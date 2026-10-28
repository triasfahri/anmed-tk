# Preprocessing approach: MBG Twitter/X corpus

Scope: this document describes how the raw collection of Indonesian posts about the free-meals programme (MBG) is turned into analysis-ready files. It covers preprocessing only. Topic modelling, engagement analysis and temporal analysis are downstream and out of scope here. The operational how-to (run commands, CLI options, column list) is in `README.md` (Indonesian). This file explains **what was done and why**.

## 1. Result in one table

| Item | Value |
|---|---|
| Raw records in | 28,081 posts (`raw_data_MBG.jsonl`), all `lang = in` |
| Period | 2025-01-01 to 2026-10-25 (UTC) |
| Invalid rows removed | 0 (no missing text, no invalid date, no duplicate id) |
| Population for every research question | all 28,081 valid posts |
| RQ1 file (`topic_documents_rq1.csv`) | 27,598 documents (483 exact duplicates collapsed; 328 groups represent 811 posts) |
| RQ2/RQ3 file (`posts_rq2_rq3.csv`) | 28,081 posts, nothing deduplicated |
| Automated checks | 19 of 19 pass |
| Manual semantic audit | mechanism provided (150-row sheet), review deliberately skipped by team decision |

## 2. Input data and provenance

- **Source.** X/Twitter posts collected with the Apify actor `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`, language filter `in`. Native retweets are excluded by the actor.
- **Keywords (7, case-insensitive):** `mbg`, `makan bergizi gratis`, `sppg`, `satuan pelayanan pemenuhan gizi`, `bgn`, `badan gizi nasional`, `embege`. The list was fixed by the researchers; it replaced an earlier automatic keyword-snowballing run whose candidates were judged unusable.
- **Collection design.** One query per keyword per calendar month (UTC), January 2025 to October 2026 (22 months, 154 keyword-month windows), at most 300 posts per window. The search returns results newest-first, but for month-wide windows it returns only a popularity-skewed subset (see limitation 1). `collection_log.csv` records the posts fetched and whether each window hit the cap.
- **Minimisation done at collection time.** Username, display name, photo and location were never stored. `@handles` in text are replaced by `@pengguna`. The author id is stored only as a salted SHA-256 hash (`user_hash`, 16 hex characters).
- **Two upstream columns that are not used:** `judge_score` and `on_topic`, produced by an IndoBERT similarity judge (centroid cosine, threshold -0.070). They stay in the raw file only and are dropped from every output (decision D1).

## 3. Pipeline at a glance

```
raw_data_MBG.jsonl
  -> load + schema check (id as string, dates to UTC, numeric engagement)
  -> sort by time (stable) so "first occurrence" means earliest post
  -> drop bio (privacy) and the unused judge columns (judge_score, on_topic)
  -> validity flags: missing text, invalid date, duplicate id
  -> four text views: text_minimal, text_embed, text_normalized, text_topic
  -> duplicate key (links ignored) + dup_group_size
  -> metadata: source_keyword, source_month, window_capped, uses_embege, has_* flags
  -> outputs:
       posts_rq2_rq3.csv           all valid posts (no text deduplication)
       posts_preprocessed.jsonl    the same posts as JSON Lines (one record per line)
       topic_documents_rq1.csv     one document per duplicate group
       manual_quality_audit.csv    150-row reproducible sample (seed 42)
       topic_stopwords.txt         stopwords for the downstream term-weighting step
       preprocessing_report.json   counts per stage and configuration
  -> validate_pipeline.py (19 checks)
```

## 4. Decisions and reasons

### D1. No relevance filter; every valid post is kept

The upstream `on_topic` flag removed 5,865 of 28,081 posts (21%). Evidence against using it as the population:

- 2,583 of the removed posts contain an explicit MBG, SPPG or BGN term; for example "MBG dari bandung hari ini #AwasiMBG" was dropped.
- On a 100-post hand-labelled sample its recall was 0.72 (precision 0.91), i.e. it loses about 28% of truly relevant posts.
- It is an embedding-similarity filter. Filtering by closeness to seed posts and then clustering embeddings would narrow topic diversity by construction.

The researchers accepted that some off-topic posts remain ("some strays are fine"), and dropped the two judge columns from all outputs so nobody filters by them by mistake. Relevance of the keyword set itself was checked on a 100-post sample of the posts that contain neither original seed term, labelled independently by three LLM annotators against a written rubric: precision 0.83 (95% CI 0.745 to 0.891), agreement kappa 0.83 to 0.93. Weakest keyword: `bgn` (17 of 24 sampled posts, precision 0.71; small sample), mainly because "bgn" is also chat slang for "bangun" (woke up). These annotators are not human; see section 7.

### D2. Two analysis files, because the questions need different duplicate handling

- **RQ1 (topics):** one document per exact-duplicate group, so one copy-pasted message cannot dominate a topic.
- **RQ2/RQ3 (engagement, time):** every post stays, because reposts are real activity and engagement belongs to each post.
- The two files are linked by `topic_document_id` (equal to `duplicate_text_group` for the representative post).

### D3. Four text views instead of one

| Column | What it is | Why it exists |
|---|---|---|
| `text` | original text | audit and reproducibility |
| `text_minimal` | HTML entities decoded, NFKC Unicode, control characters and whitespace normalised; emoji, links and mentions kept | faithful reference text |
| `text_embed` | `text_minimal` without links and without the leading run of reply mentions | recommended input for sentence embeddings: links are random strings and reply prefixes are metadata |
| `text_normalized` | lowercase, `[URL]`/`[USER]` placeholders, hashtags split into words, emoji turned into word tokens, repeated letters collapsed, slang mapped | alternative embedding input |
| `text_topic` | `text_normalized` tokens minus conservative stopwords and placeholders | term-weighting and topic labels only |

Why separate: stopword removal and slang mapping help term-based labels but remove context a sentence encoder needs. So the encoder-facing text and the label-facing text are different columns. `text_topic` must not be used as embedding input.

### D4. Cleaning rules, with the reason for each

| Step | Rule | Reason |
|---|---|---|
| Unicode and HTML | `&amp;` decoded, NFKC, control characters and runs of whitespace collapsed | same character, same token |
| URLs and mentions | `[URL]` and `[USER]` in `text_normalized`; removed in `text_embed` | content is not in the link; mentions are already anonymised |
| Hashtags | split on camelCase, underscores, digits and acronym+word (`SPPGSolo` to `sppg solo`); a 21-entry correction dictionary for frequent all-lowercase tags (`awasimbg` to `awasi mbg`) | hashtags carry the topic words; the top ~60 hashtags were reviewed by hand |
| Emoji | converted to named tokens (`emoji_nauseated_face`); skin-tone modifiers removed first so a thumbs-up with or without a skin tone is one token; ZWJ sequences kept; originals kept in `text_minimal`/`text_embed` | emoji carry complaint or sentiment signal |
| Repeated letters | runs of 3 or more collapsed to one (`telattt` to `telat`); `massa` and `Allah` stay intact | normalises emphasis without touching real double letters |
| Slang | 37-entry dictionary (`gak`, `gmn`, `udh`, `bgt`, ...); negations (`tidak`, `bukan`, `jangan`, `belum`) are kept; `dr` is deliberately **not** mapped because it also means "doctor" | conservative; each addition risks changing a domain term |
| Stopwords | 48 general Indonesian function words in `text_topic` | conservative on purpose; negations stay |
| Code-mixing | untouched, no translation | meaning of mixed posts is preserved; audit sample covers it |
| Stemming | not used | it hurts readability of topic labels and should be tested as a separate experiment |

### D5. Deduplication

- Key: lowercase `text_minimal` with links removed and whitespace collapsed. Earlier the key included each post's own `t.co` link, which hid 371 copy-paste posts.
- The earliest post of a group is the representative.
- `dup_group_size` keeps the group size in both files, so information about copy-paste volume is not lost.
- Only exact matches after this normalisation are merged. Near-duplicates (lightly edited copies) are not detected.

### D6. Query words are removed only where they would hurt

The main query words appear in 27 to 51% of documents (`mbg` 51%, `makan` 37%, `bergizi` and `gratis` 35%, `sppg` 29%, `bgn` 27%). Left alone they would head almost every topic label. They are removed **only** in the term-weighting step, through `topic_stopwords.txt` (72 words: the general stopwords, the query words, ten filler words such as `ya`, `sih`, `kok`, and the placeholder words `url`, `user`). They stay in `text_embed` and `text_topic`, so encoders keep full context. `gizi` and `program` are deliberately not stopwords.

### D7. Metadata for later analyses

- `source_keyword`, `source_month`: parsed from the stored query.
- `window_capped`: from `collection_log.csv`, true when the post's keyword-month window hit the 300-post cap (81.5% of posts; `mbg`, `makan bergizi gratis` and `bgn` are capped in every month).
- `uses_embege`: whole-word, case-insensitive match; the slang is often critical or sarcastic and may separate topics by register rather than subject.
- `has_url`, `has_mention`, `has_hashtag`, `has_emoji`, `char_count`, token counts, `type` (original 62%, reply 27%, quote 11%).

### D8. Privacy

`bio` (profile text, non-empty in about 90% of rows) is removed from every output and stays only in the private raw file. User identity exists only as `user_hash`. Usernames, names, photos and locations were never stored.

### D9. Time

All timestamps and monthly windows are UTC. A post made late in the evening on the last day of a month in Indonesia (UTC+7) can belong to the next UTC month. This is kept deliberately so that months match the collection windows and the `window_capped` flag.

## 5. Quality assurance

- **Automated (19 checks, `validate_pipeline.py`).** Raw count versus report; retention at each stage; freshness (the validator re-runs the pipeline in a temporary folder and compares hashes with `output/`, so stale files fail); behaviour cases (slang, repeated letters, skin tone, `dr`, hashtags, placeholders); deduplication fixture (earliest post kept, invalid rows excluded from group size); empty-text rules; consistency of the RQ1 file with the RQ2/RQ3 file; `bio` absence; no links in `text_embed`; group sizes; stopword file; source metadata and the capped flag against the collection log.
- **Reproducibility.** Seed 42 for the audit sample. The pipeline produces byte-identical outputs under the pinned requirements (Python 3.12, pandas 2.2.3, emoji 2.15.0) and under pandas 3.1 on Python 3.14.
- **Manual audit (mechanism only).** `manual_quality_audit.csv` has 150 reproducible rows with all four text views and empty review columns (`meaning_preserved`, `review_status`, `reviewer_notes`). The team decided to skip the semantic review and accept that some cleaning rules may change the meaning of some posts. The validator therefore reports 0 of 150 reviewed; this is expected, not a failure. Anyone who wants the check later can fill the sheet without rerunning anything.

## 6. Known limitations

1. **The collection cap breaks volume.** 81.5% of posts come from capped windows; monthly post counts are cap artefacts, and a capped window holds a popularity-skewed subset, not the newest posts: median likes are 35 in capped windows versus 3 in uncapped ones (663 for `mbg`), and a one-day check of `mbg` on 2025-01-31 found 13 Indonesian posts in six hours while the pull held 7 for the whole day. Engagement statistics from capped windows are therefore biased upward. Temporal work should use topic shares, not counts. The `window_capped` flag allows a robustness check, but the uncapped subset is a different keyword mix (mostly rare keywords), so it is a side view only.
2. **Keyword mix changes over time** (`sppg` and `embege` rise later), which can shift topic shares by itself.
3. **`bgn` ambiguity.** 7 of 24 `bgn` hits in the validation sample were not about the programme (small sample, so treat as indicative).
4. **Slang and hashtag handling is partial.** 37 slang entries and 21 hashtag corrections cover frequent cases only; less frequent hashtags are split by rule.
5. **Exact deduplication only.**
6. **Relevance evidence is LLM-labelled**, not human-labelled, and covers only posts without the two original seed terms.
7. **Long posts.** The 95th percentile post is about 175 words and the longest 2,270; truncation is a downstream decision and must follow the chosen encoder's tokenizer.
8. **Replies and quotes are included** (38% of posts, confirmed by the team); many replies are short and depend on the post they answer.
9. **No semantic audit of the cleaning rules.** Slang mapping, hashtag splitting and emoji tokens were checked by behaviour tests and by hand on frequent cases, not on a reviewed sample (see section 5).

## 7. Handoff to the next stage (interface only)

- Embedding text: `text_embed` (alternatives `text_minimal`, `text_normalized`).
- Term-based labels: `text_topic` with `topic_stopwords.txt`.
- RQ1 unit: `topic_documents_rq1.csv`; RQ2/RQ3 unit: `posts_rq2_rq3.csv`; join key: `topic_document_id`.
- `posts_preprocessed.jsonl` is the same set as `posts_rq2_rq3.csv` in JSON Lines (ids as strings, `null` for missing values, UTC ISO 8601 times). Its RQ1 document set is the rows with `is_duplicate_text == false` (27,598), identical to `topic_documents_rq1.csv`.
- Time series: use shares and the `window_capped` / `source_month` columns.

## 8. Rerun

```bash
python -m pip install -r requirements.txt     # pandas==2.2.3, emoji==2.15.0
python preprocess_twitter.py                  # writes output/
python validate_pipeline.py                   # 19 checks, writes validation_report.json
```

The raw file `raw_data_MBG.jsonl` (it contains profile bios) is not in the repository. Place it, together with `collection_log.csv`, next to the script before running.

## 9. Decisions confirmed by the team

- Manual semantic audit: skipped; the mechanism stays in the repository.
- Replies and quotes: kept (use the `type` column to filter later).
- `judge_score` and `on_topic`: dropped from all shared outputs.
- Language: this approach document stays in English; `README.md` stays Indonesian.
