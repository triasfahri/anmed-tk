# Snowball Spec: MBG keyword discovery with an IndoBERT relevance judge

Status: DRAFT for approval. Nothing is implemented. Last updated 2026-10-26.

## 1. Scope

Covers: Phase 0 smoke test, Phase 1 keyword discovery (automated snowball), Phase 2 final collection, and the human validation step.
Does not cover: BERTopic (RQ1), obstacle coding (RQ2), trend analysis (RQ3). Those start from `data/final_posts.jsonl`.

Method lineage (framework): Bozarth & Budak 2022 (seed keywords, expansion, relevance check, loop, stop on marginal gain). Two deliberate departures from that paper, both stated in the paper's method section:
1. Candidates come from corpus co-occurrence counts (your spec), not from FastText neighbours.
2. Relevance is judged automatically by a pretrained IndoBERT encoder instead of Mechanical Turk workers, and the automated judge is audited by a 100-post human sample.

## 2. Design in one picture

```
seeds ──► collect R1 ──► clean/minimize ──► posts.jsonl
                              │
        background sample ────┤
                              ▼
              calibrate judge once (frozen): μ, centroid c, threshold τ
                              │
   ┌──────────────────────────┘
   ▼
candidates (co-occurrence df) ──► pilot fetch per candidate (100 posts)
   ▼                                  │ mask keywords, embed, score vs τ
accept if precision on NOVEL posts ≥ 0.70 and ≥ 20 novel posts
   ▼
collect accepted terms ──► gain = relative growth of on-topic corpus
   ▼
stop if gain < 5% | nothing accepted | round 4     else loop
   ▼
freeze keywords_final.json ──► Phase 2 monthly collection ──► final_posts.jsonl
   ▼
validate.py: label 100 novel-only posts ──► precision (Wilson CI) + judge AUC
```

## 3. Config constants (top of `collect.py`, nothing else configurable)

| Name | Default | Note |
|---|---|---|
| `ACTOR_ID` | `"kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"` | Apify id `CJdippxWmn9uRfooo`, the actor behind `fixtures/sample_tweet.json`. Verified live 2026-10-26. Only `build_actor_input` and the query builder depend on it. |
| `SEEDS` | `["MBG", "makan bergizi gratis"]` | |
| `START`, `END` | `2025-01-01`, `2026-10-26` | Confirm with user. |
| `MAX_PER_QUERY_DISCOVERY` | 25 | Per (keyword, month window); the actor's minimum per query is 20. Spreads discovery over the whole date range (a single `Latest` query returned only the newest 6 days). |
| `PILOT_N` | 100 | Posts fetched per candidate. |
| `BACKGROUND_N` | 300 | Generic Indonesian posts, used only to calibrate the judge. |
| `BACKGROUND_QUERY` | `"yang"` with language `in` | Any very common Indonesian word. |
| `TOP_N_CANDIDATES` | 25 | |
| `MIN_LIFT` | 3.0 | Candidate must be at least this many times more frequent in seed posts than in the background sample. Heuristic. |
| `REVIEW_MIN_PRECISION` | 0.5 | Pilot pre-screen: candidates below this judge pass-rate never reach the human. |
| `REVIEW_EXAMPLES` | 3 | Example posts shown per candidate in the human review. |
| `MIN_DF` | 5 | A candidate must occur in at least this many stored posts. |
| `MAX_ACCEPT_PER_ROUND` | 5 | `--auto` mode only. |
| `MIN_PRECISION_NOVEL` | 0.70 | `--auto` mode only (no human review). Heuristic, not sourced. |
| `MIN_NOVEL_N` | 20 | Fewer novel pilot posts means "redundant", reject. |
| `STOP_GAIN` | 0.05 | Relative growth of the on-topic corpus. |
| `MAX_ROUNDS` | 4 | |
| `ENCODER` | `"indobenchmark/indobert-base-p1"` | Verified on Hugging Face (feature-extraction, MIT, Indonesian). |
| `MAX_SEQ` | 128 | |
| `MAX_PER_QUERY_FINAL` | 1000 | Per (keyword, month). |
| `PRICE_PER_1000` | 0.25 | USD, worst case. The actor charges per tweet by the account's Apify plan: FREE 0.25, BRONZE 0.22, SILVER 0.20, GOLD and above 0.18 (from the actor's pricing metadata). |

Secrets from env only: `APIFY_TOKEN`, `USER_HASH_SALT`. `APIFY_TOKEN` is a comma-separated list (`tok1,tok2,tok3`; one token also works). Tokens are shuffled once per run and used round-robin, one per paid fetch, so spend spreads across accounts. A token answering HTTP 401/402/403/429 is dropped for the rest of the run (log lines show only its position, never its text); when all are dropped the run stops and a rerun resumes. They live in `~/.config/anmedsos/env` (mode 600, outside the project). Missing means exit with a message at start of `main`, not at import. Never written to disk or logs.

## 4. Data

### 4.1 Row schema (`posts.jsonl`, `final_posts.jsonl`), exactly these keys
`id, text, created_at, likes, retweets, replies, quotes, type, parent_id, lang, user_hash, bio, followers, query, judge_score, on_topic`

Mapping from the Apify item (see `fixtures/sample_tweet.json`):
- `created_at`: parse `"Fri Sep 18 04:29:45 +0000 2026"` to ISO 8601 UTC.
- `likes/retweets/replies/quotes`: `likeCount/retweetCount/replyCount/quoteCount`.
- `type`: `quote` if `quoted_tweet` or `quoted_tweet_results` is set; else `reply` if `isReply`; else `original`.
- `parent_id`: `inReplyToId` for replies; the quoted tweet's id for quotes; else null.
- `user_hash`: first 16 hex chars of sha256(salt + author id).
- `bio`: `author.profile_bio.description` if non-empty, else `author.description`, else `""`. (In the sample, `description` is empty and `profile_bio` holds the real bio.)
- `followers`: `author.followers`.
- `query`: the query string, recovered from `searchTermIndex`.
- `judge_score`, `on_topic`: added by the judge (section 6.2).

Dropped at ingest, never written: username, display name, profile and cover picture, location, author URL, the raw author object.
Retweets (`retweeted_tweet` set) are dropped. Only `lang` in `{"in","id"}` is kept.
In `text`, every `@handle` is replaced by `@pengguna` (data minimization, supports C6). The unmasked text is never kept; keyword matching runs on the masked text.

### 4.2 Files under `data/`
- `posts.jsonl`: discovery corpus, deduped by `id`.
- `background.jsonl`: background sample (same schema, no judge fields).
- `state.json`: `{round, keywords, judge: {mu_path, centroid_path, tau, auc_r_vs_b, b0}, history: [...]}`.
- `calibration.npz`: frozen judge (`mu`, `c`, `tau`, `auc`, `b0`). No embedding cache: each post is scored once at ingestion and `judge_score` / `on_topic` are stored in its row.
- `decisions.csv`: one row per candidate per round (see 6.6). This is the audit trail for the paper.
- `keywords_final.json`: frozen keyword set (section 7).
- `final_posts.jsonl`, `collection_log.csv`: Phase 2.
- `validation_sample.csv`: Phase 3.

A rerun resumes from `state.json` and never refetches a finished step. State is rewritten after every round.

## 5. Phase 0: smoke test (about $0.004)

`python collect.py --smoke`: fetch 20 posts for `MBG`, print the keys of the first raw item, run `minimize` on all 20, print the first minimized row, exit. Purpose: confirm actor input/field names before spending money. If the field names differ from `fixtures/sample_tweet.json`, fix `build_actor_input` / `minimize` only.

`python collect.py --dry-run`: prints the round-1 queries and the cost bound, no network.

Phase 0 results (verified live 2026-10-26, about $0.015 spent):
- Output matches `fixtures/sample_tweet.json` (`searchTermIndex`, `retweeted_tweet`, `quoted_tweet`, `inReplyToId`, `author.profile_bio`), so `minimize` needed no change and `query` attribution works.
- Input field names for this actor: `searchTerms`, `maxItems`, `queryType`, `lang`. Indonesian is `lang: "in"` (the language list has no `id`). `maxItems` applies **per search term** (minimum 20) and may overshoot slightly, so `build_actor_input` splits the total across queries; the earlier worry about months starving each other does not apply.
- Native retweets are excluded by default, so no `-filter:nativeretweets` in queries.
- Date operators: plain `since:2025-03-01 until:2025-03-04` leaked hours into the next day (results ran to 2025-03-05 03:18 UTC). `since:YYYY-MM-DD_00:00:00_UTC until:YYYY-MM-DD_00:00:00_UTC` returned exactly inside the window, newest first. Queries use that form.
- Free-plan accounts see the "subscribe to a paid plan" notice in run logs; runs still succeed.

## 6. Phase 1: discovery loop

### 6.1 Round 1
Queries: one per seed per calendar month, `"{term}" since:{month start}_00:00:00_UTC until:{month end}_00:00:00_UTC` (2 seeds x 22 months = 44 queries), `lang: "in"`, `MAX_PER_QUERY_DISCOVERY` items each. Minimize, dedupe, write `posts.jsonl`. Also fetch `BACKGROUND_N` background posts into `background.jsonl`. Print cost estimate and require `y` before every paid fetch.

### 6.2 The judge (`judge.py`)
Encoder: `SentenceTransformer(ENCODER)`. This wraps the IndoBERT masked-LM with mean pooling, so the same code path also loads multilingual SBERT. `max_seq_length = MAX_SEQ`.

Text preparation for the encoder: lowercase, URLs removed, `@pengguna` kept, `#` stripped from hashtags, then **keyword masking**: delete every occurrence of the masked terms (case-insensitive, word-boundary, hashtag-aware).
Why masking: every seed-matched post contains "mbg", so an unmasked centroid mostly encodes the token "mbg". Candidates found precisely because they appear without "mbg" would then score low even when on-topic. The judge must score the surrounding context only.

Calibration (once, after round 1, then frozen for all later rounds so the judge cannot drift):
1. Embed R (round-1 posts, seeds masked) and B (background).
2. `mu` = mean of all embeddings in R ∪ B. Centering by `mu` counters the narrow-cone problem of mean-pooled masked-LM embeddings. (Common practice for raw BERT embeddings; not claimed as sourced. The validation step shows whether it works.)
3. `c` = unit-normalised mean of centered R embeddings.
4. `score(p) = cosine(e(p) - mu, c)`.
5. `tau` = threshold maximising Youden's J (TPR − FPR) separating R from B. Log `auc_r_vs_b` and `b0` (share of B above `tau`, the judge's false-positive rate on random text).
6. Abort with a clear message if `auc_r_vs_b < 0.8` (heuristic: the judge does not separate even the easy case; switch `ENCODER`). Note this AUC is optimistic because R matches the seeds by construction. It is a sanity check, not a result.

Corpus scoring: every stored post gets `judge_score` with seeds masked (fixed, so embeddings are cacheable) and `on_topic = judge_score >= tau`.

### 6.3 Candidates
From all stored posts: document frequency of hashtags, unigrams, and bigrams (each counted once per post). Preprocess: strip URLs and `@pengguna`, punctuation, digit-only tokens, tokens under 3 chars; inline stopword set (about 80 Indonesian + English words); bigrams only if neither word is a stopword. Exclude anything equal to or containing a known keyword, and word-level sub-phrases of known multi-word keywords.
Ranking (changed after the first live round, where raw df put generic words like "program", "anak", "terus" on top): `lift = (df / n_seed_posts) / ((df_background + 1) / (n_background + 1))`, using the same tokenisation on `background.jsonl`. Keep `df >= MIN_DF` and `lift >= MIN_LIFT`, sort by lift then df, top `TOP_N_CANDIDATES`. Lift measures association with the topic, not specificity to MBG, so generic-but-topical words ("sekolah", "pemerintah") can still survive; that is what the human review in 6.4 is for.

### 6.4 Pilot and precision on novel posts
For each candidate: fetch `PILOT_N` posts for the candidate alone (one actor run per candidate so `searchTermIndex` attribution is exact). Minimize. Then:
- **novel** posts are those whose text contains none of the known keywords (seeds plus accepted so far). Only novel posts are judged, because a post that already contains "MBG" adds no coverage.
- Mask known keywords and the candidate itself, embed, score.
- `precision_novel = mean(score >= tau)` over novel posts; `novel_n = len(novel)`.
- Pilot posts are used for measurement only and are discarded, never stored. Stored data comes only from the collect step, so the gain statistic in 6.5 is honest and rejected candidates leave no data behind.

One `y/N` confirm covers the whole pilot batch (cost bound = candidates x `PILOT_N`).

**Why a human decides (first live round, 2026-10-26).** The IndoBERT judge separated real seed posts from random background with held-out AUC 0.855 and passed about 27% of the background (multilingual SBERT: AUC 0.818, 24%). The background-based pass rate understates false positives for topically adjacent text: "siswa" (0.76) and "masyarakat" (0.74) cleared the old 0.70 rule although they are generic words. So the judge is evidence, not the decision.

Pre-screen: `novel_n < MIN_NOVEL_N` is rejected as redundant, `precision_novel < REVIEW_MIN_PRECISION` is rejected, the rest form the shortlist. For each shortlisted term the review screen shows `df`, `lift`, `novel_n`, the judge pass-rate, and `REVIEW_EXAMPLES` random novel pilot posts (mentions already masked). The question per term: "would a post containing only this term be about MBG?" You type the numbers to accept, and may add your own terms after `;` (added terms are accepted without a pilot). An empty answer accepts nothing and ends the loop. Your answer is saved to `state.json` before anything else happens, so a crash never re-asks. Skimming about 25 terms with 3 snippets each takes a few minutes, versus hundreds of tweets in the earlier design. This follows Bozarth & Budak, where a human judged relevance of candidate keywords.

`--auto` skips the human and uses the old rule: accept if `novel_n >= MIN_NOVEL_N` and `precision_novel >= MIN_PRECISION_NOVEL`, at most `MAX_ACCEPT_PER_ROUND`, sorted by `precision_novel` then `novel_n`. Use it only for tests and dry experiments.

Audit trail: `data/review_log.jsonl` holds one line per shortlisted term (round, term, decision, the example snippets shown, timestamp). Report in the paper how many candidates were shown and how many accepted.

### 6.5 Next round and stop rules
Round r+1 collects the accepted terms only (full cap), minimizes, dedupes, scores. Then:
`gain = (new unique posts with on_topic) / (on_topic posts in the corpus before this round)`.
Stop when any of: `gain < STOP_GAIN` (from round 2 on), no candidate accepted, `round == MAX_ROUNDS`. Print the reason. This follows the marginal-gain stopping idea named in Bozarth & Budak; the 5% value is our own and is stated in the paper.

### 6.6 `decisions.csv` columns
`round, term, df, lift, novel_n, precision_novel, tau, b0, decision, pilot_cost_usd, timestamp`
`decision` is one of `accept_human`, `reject_human`, `human_added`, `reject_precision`, `reject_redundant` (human mode) or `accept`, `reject_precision`, `reject_redundant`, `reject_cap` (`--auto`).
plus one `rounds.csv`-style history in `state.json`: `round, keywords, fetched, new_unique, new_on_topic, gain`.

## 7. Freeze
On stop, write `keywords_final.json`: `{keywords, encoder, tau, params, frozen_at}`. After this the keyword set is never edited. Any change means a new discovery run.

## 8. Phase 2: final collection
`python collect.py --final`: for each keyword in `keywords_final.json` and each month between `START` and `END`, query with that month's `since/until`. One run per keyword with all its monthly queries; `maxItems = MAX_PER_QUERY_FINAL * n_months`; attribute by `searchTermIndex`. Minimize, dedupe against each other, score with the frozen judge, write `final_posts.jsonl` with `on_topic` flagged. Do not delete off-topic rows; the analysis filters on `on_topic`.
`collection_log.csv`: `keyword, month, fetched, capped`. `capped = fetched >= MAX_PER_QUERY_FINAL` marks truncated months, which matters for RQ3. Split a capped month into weeks by hand if needed.

## 9. Phase 3: human validation (`validate.py`)
1. `python validate.py sample`: draws 100 random posts from `final_posts.jsonl` that contain none of the seed terms (the "novel stratum", where snowball risk lives). Writes `data/validation_sample.csv` with `id, text, label` (text only, no user fields). If the stratum has under 100 posts, take all.
2. You label `label = 1` if the post is about the MBG program (any aspect), else `0`. Skip unsure ones, leave blank, and count them in the report. Optional: a teammate double-codes 30 posts for Cohen's kappa.
3. `python validate.py score [--encoder NAME]` prints: precision of the novel stratum with the Wilson 95% interval, judge AUC against your labels, and precision/recall of the judge at `tau`. With `--encoder`, it recalibrates on stored round-1 and background posts using that model and rescores the same 100 labels, so comparing IndoBERT, IndoBERTweet (`indolem/indobertweet-base-uncased`), and multilingual SBERT (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, language list includes `id`) costs nothing extra. With 100 labels, treat AUC gaps under about 0.05 as noise.

## 10. Modules and functions

`judge.py`: `load_encoder`, `mask_terms`, `prep`, `auc`, `youden_threshold`, `calibrate(R, B, encode, mask)` returning a dict with `mu, c, tau, auc, b0`, `score(texts, cal, encode, mask)`, `save_calibration`, `load_calibration`.
`collect.py`: `minimize(item, salt, queries)`, `build_actor_input(queries, max_items)`, `build_queries(term, windows)`, `fetch(terms, token)` (the only Apify call; `apify_client` imported lazily inside), `extract_candidates(rows, known, top_n, min_df)`, `pilot(term, known, ...)`, `accept(results)`, `gain(...)`, `should_stop(...)`, `main()` with `--smoke`, `--dry-run`, `--final`.
`validate.py`: `sample()`, `wilson(k, n)`, `score_labels(...)`.
`test_snowball.py`: all checks below.
`requirements.txt`: `apify-client`, `sentence-transformers`, `numpy`. (sentence-transformers brings torch, transformers, scikit-learn, scipy.) Python 3.14 is installed here; if torch has no wheel for it, use a 3.12 venv.

Style: flat modules, small functions, no classes or config files, `encode` is an injectable callable so tests never download a model.

## 11. Acceptance tests (`python3 test_snowball.py` prints `ok`, no network, no model download)

Uses a fake `encode` (bag-of-words hashed to vectors) and `fixtures/sample_tweet.json`.
1. `minimize`: exactly the 16 schema keys minus judge fields at ingest; type `reply`; `parent_id == "2100182645535879648"`; `created_at == "2026-09-18T04:29:45+00:00"`; bio from `profile_bio`; `@GusDikaOfficial` becomes `@pengguna`; serialized row contains none of `myjaem13`, `putt`, `profile_images`, `profile_banners`; same salt gives same hash, different salt differs.
2. Retweet item returns None; `lang="en"` returns None; quote item gets type `quote` and the quoted id as `parent_id`.
3. `mask_terms` removes `MBG`, `#MBG`, `makan bergizi gratis` case-insensitively and leaves "mbgx" alone.
4. `calibrate` on synthetic R (food/school words) vs B (unrelated words): `auc >= 0.95`, `tau` between the two groups, `b0` small.
5. `extract_candidates` on about 8 synthetic rows: finds a repeated bigram and a repeated hashtag, excludes known term, respects `MIN_DF`.
6. Pilot logic: novel-only filtering (a post containing a known keyword is ignored), `novel_n < 20` gives `reject_redundant`, precision 0.8 accepts, 0.6 rejects, accept list capped at 5.
7. `gain` and `should_stop`: round 1 never stops on gain; round 2 gain 0.01 stops; gain 0.5 continues; round 4 stops; nothing accepted stops.
8. `wilson(80, 100)` is within 0.005 of `(0.711, 0.867)`.
9. `build_queries` output contains the since/until dates and quotes multi-word terms.
10. `python collect.py --dry-run` with dummy env vars prints queries and cost, exits 0.

## 12. Cost (USD, worst case at 0.25 per 1000 tweets, the FREE-plan tier; paid tiers go down to 0.18)

- Round 1: 2 seeds × 22 months × 25 = 1,100 posts, $0.275. Background 300, $0.075 (already fetched, reused).
- Pilots: up to 25 × 100 = 2,500 per round, $0.625, in rounds 1 to 3.
- Collect rounds 2 to 4: each accepted keyword costs 22 × 25 = 550 posts, $0.1375; say 8 accepted per round, about $1.1.
- **Phase 1 worst case: about $5.5; typical about $3.**
- Phase 2: `keywords × months × cap × 0.25/1000`. With 8 keywords × 22 months × 1000 cap, worst case about $44 (most keyword-months will fall well below the cap). With cap 2000, about $88.
- **Budget reality check:** four free accounts. If each carries Apify's usual free monthly credit (about $5, unverified, check each console), the pool is about $20, roughly 80,000 tweets at 0.25 per 1000. Phase 1 fits easily; Phase 2 needs a cap sized to the pool (for example `MAX_PER_QUERY_FINAL = 300`: 8 × 22 × 300 = 52,800 tweets, about $13). Per-run fees were not seen in the actor's pricing metadata (only a per-tweet event).

## 13. Journal support and wording

| Use | Source | Check status |
|---|---|---|
| Seed, expansion, relevance, loop framework; stopping on marginal gain; local vs other pipelines | Bozarth & Budak 2022, EPJ Data Science 11:30 | Read full text. Relevance was judged by Mechanical Turk. |
| BERT-based query expansion on Indonesian tweets; hybrid (co-occurrence + BERT) beats BERT alone | Kaban et al. 2025, Eastern-European J. of Enterprise Technologies, doi 10.15587/1729-4061.2025.340258 | Abstract read only. Checkpoint name not seen. Scopus per journal's indexing page; quartile unverified. |
| IndoBERT embeddings inside BERTopic on Indonesian X data | "Integrating IndoBERT and BIRCH of BERTopic in Indonesian short text", IJ-AI 14(5), 2025 | Abstract read only. Journal says Scopus Q2; metrics differ across sources. |
| IndoBERT | Wilie et al. 2020 (IndoNLU) and Koto et al. 2020 (IndoLEM) | Conference papers, not journals. |
| IndoBERTweet (swap-in) | Koto et al. 2021, EMNLP | Conference paper. |
| King et al. 2017, AJPS 61(4) | Computer-assisted keyword discovery; human in the loop | Abstract read only. |

Honest gap: no journal found tests a pretrained encoder as the relevance step in a snowball loop. The 100-post validation carries that claim. Counter-evidence to mention in limitations: Diaz et al. (ACL 2016) found locally trained embeddings beat global ones for query expansion (static vectors, retrieval task).

Draft method paragraph (Indonesian, fill the numbers):
> Kata kunci tambahan ditentukan secara iteratif dengan kerangka Bozarth dan Budak [10]: kata kunci awal digunakan untuk menjaring unggahan, kandidat diekstraksi berdasarkan frekuensi kemunculan bersama, dan hanya kandidat yang relevan yang dimasukkan ke putaran berikutnya. Berbeda dari [10] yang memakai penilai manusia (Amazon Mechanical Turk), relevansi dinilai otomatis dengan representasi kalimat IndoBERT [x] yang dibandingkan terhadap centroid unggahan kata kunci awal dengan kata kunci dimasking, sejalan dengan ekspansi kueri berbasis BERT pada tweet berbahasa Indonesia [y]. Ambang batas ditetapkan otomatis (Youden's J) dan diaudit pada 100 unggahan yang dilabeli manual (presisi X%, CI 95% [a, b]). Iterasi berhenti ketika pertumbuhan relatif korpus relevan kurang dari 5% [10].

## 14. Risks and limits

1. **IndoBERT is a masked LM, not a sentence encoder.** Mean pooling plus centering and the auto threshold are mitigations; validation shows whether they suffice. If AUC against your labels is weak, swap `ENCODER` (one constant).
2. **Code-switching.** IndoBERT's vocabulary is Indonesian; English chunks split into subwords and may embed noisily. This is the main reason to run the encoder swap test in section 9.
3. **Round-1 centroid is narrow** (seed-matched posts only), so novel subtopics may score low. Effect: conservative, lower recall. Judge is frozen on purpose to avoid drift.
4. **Masking** removes keyword signal by design; very short posts can end up near-empty and score unpredictably. Posts under 3 tokens after masking get no score and count as off-topic.
5. **Actor drift / pricing**: field names and any per-run fee are unverified until Phase 0.
6. **X data ethics**: public posts, minimized fields, hashed IDs, mention masking. Check the campus ethics requirement for storing bios.
7. Language filter `in` drops posts X mislabels as other languages; some code-mixed posts are lost.

## 15. Open decisions

1. Date range (`START`, `END`).
2. (Done 2026-10-26) Actor confirmed and Phase 0 run, see section 5.
3. Phase 2 cap (`MAX_PER_QUERY_FINAL`) and total budget.
4. Who labels the 100 posts; whether a teammate double-codes 30.
5. Python environment (3.12 venv if torch install fails on 3.14).

## 16. Implementation status (2026-10-26)
Implemented and offline-tested (`../.venv/bin/python test_snowball.py` prints `ok`): `judge.py`, `collect.py`, `validate.py`, `requirements.txt`.
Not yet run: anything against the real Apify actor, anything with the real IndoBERT model (needs `pip install sentence-transformers`, about 2 GB with torch, plus a 500 MB model download on first use).

Decision (2026-10-26): `ENCODER` stays `indobenchmark/indobert-base-p1`. Real-model probe on 14 hand-written sentences: IndoBERT and `paraphrase-multilingual-mpnet-base-v2` both AUC 1.000; SBERT had more margin (d' 3.62 vs 3.29, hard negative cleared by 0.10 vs 0.001). Too small to decide, so the choice is settled on the 100 labeled posts: run `validate.py score` (IndoBERT) and `validate.py score --encoder sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, and report both. Real environment: torch 2.14.1 + sentence-transformers 6.2.0 install fine on Python 3.14, MPS available.

Run order:
1. `set -a; source ~/.config/anmedsos/env; set +a` (`APIFY_TOKEN=tok1,tok2,...`, `USER_HASH_SALT=...`; pick the salt once and keep it, changing it breaks user-hash continuity).
2. `../.venv/bin/python collect.py --dry-run`, then `--smoke` (both already verified 2026-10-26).
3. `../.venv/bin/python collect.py` (discovery; prompts before every paid fetch, `--yes` skips), then `--final`.
4. `../.venv/bin/python validate.py sample`, label `data/validation_sample.csv`, then `validate.py score [--encoder NAME]`.
