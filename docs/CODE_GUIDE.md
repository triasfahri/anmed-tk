# Code guide

Audience: anyone who must run, check or explain the code. The reasons behind the design are in [APPROACH.md](APPROACH.md). Every script is plain Python with small functions; no framework.

## 1. Repository map

```
tugas-kelompok/
├── collect.py              batch 1: monthly searches (also shared helpers)
├── extra_pull.py           batch 2 PLANNER: dry run only, spends nothing
├── run_extra_pull.py       batch 2 EXECUTOR: runs the plan, spends credit
├── test_collect.py         offline tests for collect.py
├── test_extra_pull.py      offline tests for the planner
├── test_run_extra_pull.py  offline tests for the executor
├── requirements.txt        apify-client
├── fixtures/sample_tweet.json   one real raw item, used by the tests
├── preprocessing/
│   ├── merge_raw.py            joins batch 1 and batch 2 into one raw file
│   ├── preprocess_twitter.py   the preprocessing pipeline
│   ├── validate_pipeline.py    22 automated checks
│   ├── collection_log.csv      which keyword-months hit the 300 cap (counts only)
│   ├── requirements.txt        pandas==2.2.3, emoji==2.15.0
│   ├── README.md               operational guide (Indonesian)
│   └── PREPROCESSING_APPROACH.md   cleaning decisions in detail (English)
├── docs/                   this guide and APPROACH.md
└── data/                   collected data and logs (NOT in git)
```

Everything under `data/`, plus `preprocessing/raw_data_MBG*.jsonl`, `preprocessing/output/` and `preprocessing/.vendor/`, is **gitignored on purpose**: the repository is public and the raw files contain profile bios. See section 8.

## 2. Setup

- **Collection scripts:** `pip install -r requirements.txt` (only `apify-client`). We ran them under Python 3.14 in a shared virtual environment; other versions were not tested.
- **Preprocessing:** `cd preprocessing && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. We tested under Python 3.12 with the pinned pandas 2.2.3 and emoji 2.15.0. A vendored copy of `emoji` in `preprocessing/.vendor/` is used automatically if present.
- **Secrets (collection only):** two environment variables.
  - `APIFY_TOKEN`: one or more Apify tokens separated by commas, for example `token1,token2,token3`. The scripts rotate through them and never print them.
  - `USER_HASH_SALT`: a secret string used to hash author ids. **Use the same salt for every collection**, otherwise the same author gets different hashes.
  - We keep them in `~/.config/anmedsos/env` (outside the repo) and load them with `set -a; source ~/.config/anmedsos/env; set +a`. Never commit them.

## 3. Run order, start to finish

```bash
# --- Collection (spends Apify credit; run from tugas-kelompok/) ---
python collect.py --dry-run                 # prints query count and cost bound, spends nothing
python collect.py --smoke                   # tiny paid test (20 posts)
python collect.py --final --yes             # batch 1: 7 keywords x 22 months, resumable

python extra_pull.py                        # batch 2 PLAN: reads Apify usage (read-only), writes the plan
python run_extra_pull.py --yes              # batch 2 RUN: spends credit; add --chunks 19 to run one chunk

# --- Preprocessing (free; run from tugas-kelompok/preprocessing/) ---
python merge_raw.py                         # data/final_posts.jsonl + data/extra_posts.jsonl -> raw_data_MBG.jsonl
python preprocess_twitter.py                # writes output/
python validate_pipeline.py                 # 22 checks, rewrites output/validation_report.json
```

Tests (all offline, no network, no credit; each prints `ok`):

```bash
python test_collect.py && python test_extra_pull.py && python test_run_extra_pull.py
```

## 4. Collection code

### `collect.py` (batch 1 and shared helpers)
- **What it does:** builds one search per keyword per month, asks the actor for up to 300 posts each, converts raw items to our slim format, appends them to `data/final_posts.jsonl`, and records per keyword-month how many items came back and whether the cap was hit (`data/collection_log.csv`).
- **Key pieces**
  - `KEYWORDS`, `START`/`END`, `MAX_PER_QUERY_FINAL` (300), `PRICE_PER_1000` (0.25): the settings that define batch 1.
  - `month_windows` and `build_queries`: make the monthly search strings (`mbg since:2025-01-01_00:00:00_UTC until:2025-02-01_00:00:00_UTC`).
  - `build_actor_input`: the actor takes `maxItems` per search term with a minimum of 20.
  - `fetch`: runs the actor and returns the items. A run stopped only by the platform cost cap still returns its items.
  - `rotating_fetch`: shuffles the tokens, rotates through them, and drops a token that gets an error 401/402/403/429.
  - `minimize`: **the privacy step.** Drops retweets and non-Indonesian items; replaces `@handles` with `@pengguna`; hashes the author id with the salt (first 16 hex characters of SHA-256); drops username, display name, photo and location. It keeps the profile bio (the preprocessing step removes it from outputs).
  - `final`: loops over keywords; **skips a keyword whose months are all already logged**, so it is safe to rerun after a crash; removes ids already seen.
- **Safety:** prints the cost bound before every paid call and asks `Continue? [y/N]` unless `--yes`.

### `extra_pull.py` (batch 2 planner)
- **What it does:** reads `data/collection_log.csv` (which keyword-month cells hit the cap) and creates the plan for the short-window pull. **It has no fetching code.** It writes `data/extra_pull_plan.json` (the machine-readable plan) and `data/extra_pull_plan.csv` (one row per keyword-month).
- **Key pieces**
  - `sample_slots(month, slots_per_cell)`: the 6-hour windows. Window days are `floor((k + 0.5) * L / D)` for k = 0..D-1 (evenly spaced over a month of L days); the start hour is `(phase + k * 5) mod 24` with `phase = month_index * 7 mod 24`, so hours rotate within a month and differ between months. Self-checks assert even spacing, distinct starts and hour coverage.
  - `build_query`: keyword, then ` -earlier_keyword` exclusions, then `since_time:<epoch> until_time:<epoch>`.
  - `make_chunks`: groups up to 6 cells of one keyword into one future actor run.
  - `assign_chunks`: gives each run to the token with the most free credit, keeping a reserve.
  - `choose_plan`: `--slots-per-cell auto` picks the largest number that fits the budget.
- **Flags:** `--slots-per-cell` (default 12), `--items-per-slot` (default 20, minimum 20), `--tokens` (1-based positions in `APIFY_TOKEN`, default `1,3,4`; we skipped account 2 because it was nearly empty), `--remaining` (credit per token, skips the Apify lookup), `--reserve` (default $0.20 kept per token).
- **Cost model:** `price = $0.25/1000`; a safety factor of 1.10 allows for the actor returning slightly more than asked.

### `run_extra_pull.py` (batch 2 executor)
- **What it does:** runs the chunks of the plan one after another.
- **Safety features** (these matter because it spends real credit)
  - Refuses to run without `--yes`.
  - Checks every chunk against the plan (token position exists, bounds cover the cost).
  - **Credit guard:** before each chunk it reads the account's remaining credit and skips the chunk if it is below the chunk's safe cost plus $0.02.
  - **Resume log** `data/extra_pull_log.csv`: a chunk already logged as `ok` with the same queries is skipped on rerun, so nothing is paid for twice. Failed or skipped chunks are retried.
  - Token strings are removed from every error message before it is logged or printed.
- **Per kept post** it adds `pull_batch = "slot_sample"`, `source_keyword`, `source_month`, `slot_since`, `slot_until`, then appends to `data/extra_posts.jsonl`.
- **Counts per chunk** are logged: items returned, kept, duplicates of posts we already hold, dropped for language/retweet, dropped because outside the window (0 in our run), and posts that contained an excluded keyword (0 in our run).
- **Never touches** `data/final_posts.jsonl`.

## 5. Preprocessing code (in `preprocessing/`)

### `merge_raw.py`
Joins batch 1 (`final_posts.jsonl`) and batch 2 (`extra_posts.jsonl`) into `raw_data_MBG.jsonl`. Adds `pull_batch = "month_wide"` to batch-1 rows. Refuses to write if an id appears twice. The first time it runs it renames the old `raw_data_MBG.jsonl` to `raw_data_MBG_batch1_month_wide.jsonl` (never overwrites or deletes). Running it again produces the identical file.

### `preprocess_twitter.py`
Reads `raw_data_MBG.jsonl` and writes five files into `output/`. Main steps, in order:
1. `load_data`, `prepare_data`: read the file, keep ids as strings, parse dates to UTC, drop `bio`, `judge_score` and `on_topic`, sort by time so "first" means earliest.
2. Cleaning functions create the four text columns (see section 6): `clean_minimal`, `clean_embed`, `clean_normalized`, `clean_topic`; helpers `normalize_unicode`, `replace_social_elements`, `segment_hashtag`, `process_emoji`, `normalize_repetitions`, `normalize_slang`.
3. Duplicate key: lowercase text without links, hashed; `dup_group_size` counts posts per key.
4. Metadata: `source_keyword`, `source_month` (kept from batch 2, parsed from `query` for batch 1), `window_capped` (looked up in `collection_log.csv`), `pull_batch`, `uses_embege`, and `has_*` flags.
5. `build_outputs`: all valid posts for the posts file; one document per duplicate group for the topic file.
6. `make_audit_sample` (150 rows, seed 42), `write_jsonl`, the report, and `topic_stopwords.txt`.

**Flags:** `--input`, `--collection-log`, `--posts-output`, `--topic-output`, `--audit-output`, `--report`, `--stopwords-output`, `--json-output`, `--audit-size`, `--seed`. Defaults are relative to the script folder, so it works from any directory.

### `validate_pipeline.py`
Runs 22 checks and writes `output/validation_report.json`. In plain words they cover: the code compiles; the output files are fresh (the validator reruns the whole pipeline in a temporary folder and compares fingerprints, so stale files fail); row counts match the report; each cleaning rule behaves as intended (for example `telattt` becomes `telat`, `massa` stays `massa`, `dr` is not changed to `dari`, hashtags split correctly, links and mentions removed where required); the duplicate rules pick the earliest post; no profile bio or judge column appears in any output; every output text is non-empty; the topic file is a consistent subset of the posts file; `source_keyword`, `source_month`, `window_capped` and `pull_batch` are correct and match the raw file and the collection log; the JSONL file equals the CSV. It also reports how many of the 150 audit rows were reviewed (0, by decision).

## 6. Data files and columns

### Files

| File | Made by | What it holds | In git? |
|---|---|---|---|
| `data/final_posts.jsonl` | `collect.py` | batch 1, 28,081 posts | no |
| `data/collection_log.csv` | `collect.py` | per keyword-month: items fetched, capped yes/no | no (a copy is in `preprocessing/`) |
| `data/extra_pull_plan.json/.csv` | `extra_pull.py` | the batch-2 plan | no |
| `data/extra_posts.jsonl` | `run_extra_pull.py` | batch 2, 7,633 posts | no |
| `data/extra_pull_log.csv` | `run_extra_pull.py` | one row per executed chunk (counts, cost estimate, status) | no |
| `data/archive/` | manual | old runs and files kept for reference | no |
| `preprocessing/raw_data_MBG.jsonl` | `merge_raw.py` | both batches, 35,714 posts, with bios | no |
| `preprocessing/raw_data_MBG_batch1_month_wide.jsonl` | `merge_raw.py` | untouched batch 1 | no |
| `preprocessing/output/posts_rq2_rq3.csv` and `posts_preprocessed.jsonl` | `preprocess_twitter.py` | all 35,714 posts, same content in two formats | no |
| `preprocessing/output/topic_documents_rq1.csv` | `preprocess_twitter.py` | 35,155 distinct-text documents | no |
| `preprocessing/output/manual_quality_audit.csv` | `preprocess_twitter.py` | 150-row review sheet (empty review columns) | no |
| `preprocessing/output/preprocessing_report.json`, `validation_report.json`, `topic_stopwords.txt` | pipeline / validator | counts per stage, configuration, check results, stopword list | no |
| `preprocessing/collection_log.csv` | copied from `data/` | counts only, no posts | yes |

### Main columns of the output files

| Column(s) | Meaning |
|---|---|
| `id`, `created_at` | post id (text) and UTC time |
| `likes`, `retweets`, `replies`, `quotes`, `followers` | engagement and author size |
| `type` | `original`, `reply` or `quote`; `parent_id` is the post replied to or quoted |
| `user_hash` | pseudonymous author id |
| `text` | original text (handles already replaced by `@pengguna`) |
| `text_minimal` | original with only technical cleaning |
| `text_embed` | no links, no leading reply mentions; **input for language-model embeddings** |
| `text_normalized` | lowercase, slang mapped, emoji turned into words, placeholders for links and mentions |
| `text_topic` | normalized text without stopwords; **input for counting words and labelling topics** |
| `duplicate_text_group`, `dup_group_size`, `is_duplicate_text`, `topic_document_id` | duplicate handling; `is_duplicate_text == false` rows are the topic documents |
| `pull_batch`, `slot_since`, `slot_until` | which batch; the 6-hour window (batch 2 only) |
| `source_keyword`, `source_month`, `window_capped` | keyword and month cell the post was found in; whether that cell hit the cap |
| `uses_embege`, `has_url`, `has_mention`, `has_hashtag`, `has_emoji`, `char_count`, `token_count_*` | convenience flags and lengths |
| `query` | the exact search that found the post |

## 7. Things that went wrong, so you do not repeat them

- **Reading JSON lines:** tweets can contain the character U+2028, which Python's `splitlines()` treats as a line break and cuts posts in half. All code reads with `text.split("\n")`.
- **Language setting:** the actor's `lang` input does not filter. Dropping non-Indonesian items happens in `minimize`.
- **Date bounds:** use `since:`/`until:` only at midnight (`_00:00:00_UTC`); use `since_time:`/`until_time:` epoch seconds for anything inside a day.
- **Long windows return popular posts only** (APPROACH.md section 5.3). Prefer short windows.
- **Apify usage lags:** the account usage number can take minutes to update after a run, so it cannot be used for exact per-run cost; the scripts estimate cost from items returned.
- **Credit guard thresholds assume a snapshot** of remaining credit; leave the reserve in place.
- **A run that hits an account's limit stops early** and can return fewer items than planned. The executor's credit guard avoids this; if you change the plan, keep the guard.
- **Report paths:** `preprocessing_report.json` stores absolute paths, so after moving the folder, rerun the pipeline before running the validator (the freshness check would otherwise fail).

## 8. Privacy and sharing

- The repository is **public**. It must contain only code, documentation and the small `collection_log.csv`.
- Do **not** commit anything in `data/` or the raw/output files in `preprocessing/`. The raw files include profile bios; all files include post text from real accounts.
- Before sharing data with anyone outside the team, check the platform's terms and your institution's data rules. We did not research those (**unverified**).
- Outputs never contain bios, usernames, display names, photos or locations; author identity exists only as `user_hash`.

## 9. How to change things safely

- **Add or change a keyword:** edit `KEYWORDS` in `collect.py`, run `collect.py --dry-run` to see the cost bound, then `--final`. New keywords go through the same monthly design. Keep the keyword order in mind: later keywords exclude earlier ones in batch 2.
- **Change the time range:** edit `START`/`END` in `collect.py`; months already in `data/collection_log.csv` are skipped.
- **Spend more on batch 2:** run `extra_pull.py` again with different `--slots-per-cell`, check the printed cost, then `run_extra_pull.py --yes`. A new plan has new query hashes, so old chunks are not skipped; check the log first.
- **Change a cleaning rule:** edit `preprocess_twitter.py`, rerun it, then run `validate_pipeline.py`; add a behaviour check there for the new rule.
- **Rerun everything from raw files:** `merge_raw.py`, `preprocess_twitter.py`, `validate_pipeline.py`. The result is deterministic.
