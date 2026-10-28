# Approach and rationale

Audience: the team writing the project report. This document explains **what we did, why we did it, what else we considered, and what the data can and cannot support**. It is written in plain language. The code itself is explained in [CODE_GUIDE.md](CODE_GUIDE.md); the cleaning rules in detail are in [../preprocessing/PREPROCESSING_APPROACH.md](../preprocessing/PREPROCESSING_APPROACH.md).

Numbers in this file were measured on the final data (2026-10-28). Anything we could not verify is marked **unverified**.

## 1. The 60-second version

- **Goal:** collect public X/Twitter posts about Indonesia's free-meals programme (MBG, Makan Bergizi Gratis) and prepare them for three research questions (topics, obstacle categories with engagement, trend over time).
- **Data:** 35,714 Indonesian posts, 2025-01-01 to 2026-10-28, found with 7 fixed keywords. They come in **two batches** collected in different ways (section 5).
- **Preprocessing:** a reproducible Python pipeline turns the raw posts into clean text columns, removes exact duplicate texts for topic modelling (35,155 documents), removes personal fields, and passes 22 automated checks.
- **Biggest caveats (read section 8):**
  1. Monthly post counts are **not** a measure of how much people talked, because we capped collection per keyword and month.
  2. Batch 1 is skewed toward popular posts; batch 2 is the fairer sample for engagement.
  3. The keyword relevance check was done by AI annotators, not humans.
  4. Topic modelling, manual coding and the semantic audit of the cleaning rules have **not** been done.
- **Cost:** about $15.2 of the $20 free Apify credit (4 free accounts); $4.8 is left and expires on 2026-11-25.

## 2. Research questions the data is built for

(As understood from the project brief; confirm the exact wording with the assignment.)

| RQ | Question | What it needs from the data |
|---|---|---|
| RQ1 | Which topics appear in the discussion of MBG implementation? | A set of distinct documents; clean text for topic modelling (BERTopic was the planned method). |
| RQ2 | What obstacle categories appear, and how much engagement do they get? | Post text, likes/retweets/replies/quotes, a way to judge popularity bias. |
| RQ3 | How does the discussion change over time? | Timestamps and a fair view of each month. |

## 3. How the pieces fit together

```
7 fixed keywords
   |
   v
Batch 1: one search per keyword per month, up to 300 posts each      -> 28,081 posts
   |   (probes then showed this batch is skewed toward popular posts)
   v
Batch 2: 12 short 6-hour windows per busy keyword-month, 20 posts each -> 7,633 posts
   |
   v
Merge into one raw file, every post labelled with its batch (pull_batch)  -> 35,714 posts
   |
   v
Preprocessing: clean text views, duplicate handling, labels, privacy
   |
   v
posts file (35,714)  +  topic documents file (35,155)  +  checks/report
```

Plain names: **month_wide** in the data means batch 1 (monthly searches). **slot_sample** means batch 2 (short 6-hour windows). A "slot" is just one 6-hour window.

## 4. Choosing the keywords

**What we use (fixed list, case-insensitive):** `mbg`, `makan bergizi gratis`, `sppg`, `satuan pelayanan pemenuhan gizi`, `bgn`, `badan gizi nasional`, `embege`.

**Why these:** people write the programme's name in different forms: the abbreviation (MBG), the full name, the name of the agency that runs it (BGN, Badan Gizi Nasional), the kitchen unit (SPPG, Satuan Pelayanan Pemenuhan Gizi) and a slang spelling (`embege`). One keyword alone would miss a large part of the conversation.

**What we tried first and dropped: automatic keyword discovery ("snowballing").**
The idea (from the literature on keyword-based social media collection) is to start with seed words, collect posts, find words that co-occur, test whether they are relevant, and repeat. We built it, using a pretrained Indonesian language model (IndoBERT) as an automatic relevance judge so nobody had to read hundreds of posts. It was dropped because:
- The judge was too weak on real posts: about 25% of generic background posts passed as "relevant", and generic words like "siswa" (students) and "masyarakat" (society) were accepted.
- Held-out accuracy (AUC) was 0.855 for IndoBERT and 0.818 for a multilingual sentence model; good on a clean test, not good enough to steer an automatic loop on messy tweets.
- The team decided to choose the keywords itself. The code is still in the git history (commit `72e33a2`) if the report wants to describe it as a rejected approach.

**How good is the final list?** We labelled a random sample of 100 collected posts that contain neither "MBG" nor "makan bergizi gratis" (the hardest cases, found only through the other keywords). **83 of 100 were about the programme** (95% interval 74.5% to 89.1%). By keyword: `sppg` 27/31, `embege` 28/35, `badan gizi nasional` 5/6, `bgn` 17/24 (the weakest: "bgn" is also slang for "bangun", woke up), `satuan pelayanan pemenuhan gizi` 2/2, `mbg` 1/2 (the last two are too small to read). The labelling was done by **three AI annotators** (Codex, Claude, and a second Claude that never saw the others' labels) against a written rule. They agreed well (kappa 0.83 to 0.93). **No human labelled this sample.** The report should say so.

## 5. Collecting the data, and why there are two batches

### 5.1 Source and constraints
- **Tool:** an Apify "actor" that scrapes X search and charges per post returned (`kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`). We chose a pay-per-result scraper because it fit a small budget. We did not evaluate the official X API (**unverified** whether it would have been cheaper or allowed).
- **Money:** 4 free Apify accounts with $5 each per month; price about $0.25 per 1,000 posts (measured: 80 posts cost $0.02). The accounts' credit resets monthly.
- **Actor behaviour that shaped the design** (all found by testing):
  - You cannot ask for fewer than 20 posts per search term, so tiny searches still cost a minimum.
  - Its language setting does **not** filter language: roughly a third of returned items are non-Indonesian and are dropped by us (but still paid for).
  - It returns native retweets only if asked; we do not ask.
  - Date bounds of the form `since:YYYY-MM-DD_HH:MM:SS_UTC` work only at midnight; times inside a day return nothing. Bounds given as Unix epoch seconds (`since_time:` / `until_time:`) work down to the hour.
  - Searching a long window (a month) returns only a **popular subset** of the matching posts (section 5.3).

### 5.2 Batch 1: monthly searches (28,081 posts)
- **Design:** for each of the 7 keywords and each of the 22 months (2025-01 to 2026-10, UTC), one search asking for at most 300 posts. That is 154 keyword-month "cells".
- **Why monthly windows and a cap of 300:** it spreads coverage evenly over the 22 months and keeps the maximum cost known in advance ($11.55 worst case). One big search per keyword would have been dominated by the most recent weeks.
- **Result:** 98 of the 154 cells hit the 300 cap ("capped"); 56 returned fewer, which means those cells are complete.
- **Cost:** about $9.3 for the posts, plus about $2.8 spent earlier on the discarded keyword-discovery attempt and tests.

### 5.3 What went wrong: the popularity skew
While planning more collection we ran small paid probes (about $0.11 in total) and found that batch 1 is **not** a plain sample:
- Median likes per post were **35 in capped cells versus 3 in complete cells**; for `mbg` the median was **663 likes**.
- A one-month search for `mbg` in January 2025 returned exactly the 268 Indonesian posts we already had, in strict newest-first order but with big gaps (for example 22:56, then 16:41, 12:54, 11:30). A one-day search for 2025-01-31 returned a dense run of ordinary posts (median about 6 likes); of 13 Indonesian posts in six evening hours, 12 were missing from batch 1.
- Conclusion: for long windows the search backend returns popular posts. This is observed behaviour; **we do not know the exact mechanism** (unverified).

**Why it matters:** any engagement result (RQ2) from batch 1 alone would be inflated, and the topics would lean toward posts that went viral.

### 5.4 Batch 2: short 6-hour windows (7,633 posts)
- **Design:** inside each of the 98 capped cells, 12 windows of 6 hours, 20 posts each. Window days are spread evenly over the month and the start hour **rotates** through the day, so every hour of the day is covered about equally.
- **Why short windows:** short windows return ordinary posts, not just popular ones (median 5 likes overall, 11 for `mbg`).
- **Why rotating hours:** Indonesian daytime posting is about four times the night rate. Taking "the newest posts of each day" would sample mostly the late-UTC night hours. We saw this directly: 20 posts covered 1.5 hours in the afternoon but 6 hours at night.
- **Why exclude earlier keywords in each search:** a post that mentions both MBG and SPPG would be returned by both searches and paid for twice. Each search after the first excludes the earlier keywords (for example `bgn -mbg -"makan bergizi gratis" -sppg`). Consequence: a post is attributed to the **first** keyword (in the order mbg, makan bergizi gratis, sppg, bgn, badan gizi nasional, embege, satuan pelayanan pemenuhan gizi) that matches it. Rare keywords therefore have few batch-2 posts (`satuan pelayanan pemenuhan gizi`: 0, `badan gizi nasional`: 29), because nearly every post using the full names also says MBG or SPPG.
- **Why 12 windows of 20:** it spread the remaining budget evenly over the 98 cells (planned worst case $5.88). Real cost was **$2.95** because many windows held fewer than 20 posts.
- **Result:** 11,709 items came back; 7,633 were kept; 1,322 were duplicates of posts we already held; 2,754 were non-Indonesian or retweets. No kept post fell outside its window.
- **Not done:** a second round of windows with the remaining credit on the busiest cells (`mbg`, and the 2026 months of `sppg`, `bgn`, `embege`).

### 5.5 Why the batches are kept apart (the `pull_batch` column)
The two batches are biased in different ways (popular-skewed versus short-window clusters). If they were mixed without a label, results would depend on how many posts each batch happened to contribute, and nobody could undo it later. The label costs nothing and removes no data.

| Question | How to use `pull_batch` |
|---|---|
| RQ1 topics | Use all documents; ignore the label. Optionally check that topics are similar in both batches. |
| RQ2 engagement | Compare the batches or use batch 2 as the less biased one. Do not pool them blindly. |
| RQ3 time | Use each topic's share of a month (not counts), and check that both batches show the same trend. |

## 6. Preprocessing: what and why (summary)

Full detail and the evidence table are in [PREPROCESSING_APPROACH.md](../preprocessing/PREPROCESSING_APPROACH.md). The key decisions:

| Decision | Why |
|---|---|
| **Keep every valid post.** An earlier automatic "on-topic" filter was dropped. | It removed 21% of posts, 2,583 of which contained an explicit MBG, SPPG or BGN term. On our 100-post sample it kept only 72% of the truly relevant posts. It would also bias topics toward the seed posts. We accept some off-topic posts ("strays"). |
| **Two analysis files** (all posts; and one document per distinct text). | Topic modelling should not let one copy-pasted message dominate. Engagement and time analysis need every post. |
| **Four text versions** (`text_minimal`, `text_embed`, `text_normalized`, `text_topic`). | Cleaning that helps counting words (lowercase, slang mapping, stopwords) hurts language models that need full context. Separate columns let each step use the right one. |
| **Remove links and leading reply mentions for embeddings.** | They are random strings or metadata that add noise. |
| **Detect duplicates ignoring links.** | Reposted headlines differ only by their own short link. |
| **Query words are removed only when counting words for topic labels.** | The main query words appear in roughly 10% to 50% of documents and would head every topic label. |
| **No stemming, no translation, light slang dictionary (37 entries).** | Keeps topic labels readable and code-mixed posts intact. |
| **Privacy:** drop profile bios from outputs; never store usernames, names, photos or locations; hash user ids with a secret salt; replace @handles with `@pengguna`. | The data is about real people. Bios are free text and can identify them. |
| **All times in UTC.** | Collection windows are UTC, so months and flags match. Indonesian local time (UTC+7) would shift late-evening posts to the next day. |

## 7. Quality assurance

- **Automated:** 22 checks in `preprocessing/validate_pipeline.py` plus 46 offline unit tests for the collection code. The validator reruns the whole pipeline and compares file fingerprints, so stale output fails.
- **Reproducibility:** the pipeline gave byte-identical output on two Python/pandas versions (verified before the batches were merged; after the merge only the pinned versions were run). The audit sample uses a fixed seed (42).
- **Not done:** the 150-row sheet for checking whether cleaning changed meanings is provided but was **deliberately skipped** by the team. Slang mapping, hashtag splitting and emoji tokens were checked by tests and by hand on frequent cases only.

## 8. Limitations the report must state

1. **Monthly counts are not volume.** 85% of posts (30,525) come from capped cells; the cap, not public interest, decides the monthly count. Use shares, not counts. (Monthly totals range from 853 to 2,368 posts; that range reflects the cap, not real attention.)
2. **Popularity skew in batch 1** (section 5.3). Engagement from batch 1 is biased upward.
3. **Batch 2 is clustered**: in a busy window the 20 posts are consecutive, so each window is a small cluster, not independent posts.
4. **Keyword mix changes over time** (`sppg` and `embege` rise later), which can move topic shares by itself.
5. **`bgn` is ambiguous** (slang for "bangun"); 7 of 24 sampled `bgn` posts were unrelated.
6. **Relevance evidence is AI-labelled** and covers only posts without the two original seed terms.
7. **Language filtering is imperfect:** items were dropped by the language tag the platform gave; short or mixed-language posts may be mislabelled.
8. **Exact duplicates only.** Lightly edited copies stay separate.
9. **Replies and quotes are included** (about 43% of posts). Many replies are short and depend on the post they answer.
10. **The last month is partial.** Batch 1 ends 2026-10-25; batch 2 reaches 2026-10-28 (84 posts after the 25th).
11. **No semantic audit** of the cleaning rules (section 7).
12. **Nothing has been modelled yet.** No topics, no obstacle coding, no trend analysis exist.

## 9. What the report can and cannot say

**Can say**
- Posts were collected with 7 researcher-chosen keywords over 22 months using a pay-per-result scraper, in two batches, and merged with a batch label.
- Automatic keyword discovery was tried and rejected because the language-model judge was too unreliable on tweets.
- Keyword relevance was about 83% on a hard-case sample, labelled by AI annotators with high agreement.
- Personal data was minimised (hashed ids, no usernames or bios in outputs).

**Cannot say**
- "Discussion peaked in month X" based on post counts.
- "Posts about topic T get N likes" from batch 1 alone without mentioning the popularity skew.
- "This is a complete or random sample of all MBG posts on X." It is not.
- Anything about topics or obstacles yet.

**AI assistance:** the code and parts of the documentation were produced with AI coding assistants (Claude Code and Codex) under the team's direction. Disclose this as your course requires.

## 10. Facts and figures (one place for the report)

| Item | Value |
|---|---|
| Total posts | 35,714 (batch 1: 28,081; batch 2: 7,633) |
| Topic documents (RQ1) | 35,155 (559 exact duplicate texts removed; 25 duplicate groups span both batches) |
| Period | 2025-01-01 to 2026-10-28 (UTC) |
| Post types | original 56.6%, reply 31.6%, quote 11.8% |
| Posts with emoji / link / hashtag | 21.5% / 43.0% / 18.2% |
| Median likes | batch 1: 19, batch 2: 5 (`mbg`: 663 vs 11) |
| Capped / complete cells | 98 / 56 of 154 |
| Posts from capped cells | 30,525 (85%) |
| Money spent | about $15.2 of $20 (batch 2: $2.95; probes: $0.11) |

Posts by keyword and batch:

| Keyword | Batch 1 | Batch 2 | Total |
|---|---|---|---|
| mbg | 5,252 | 3,403 | 8,655 |
| makan bergizi gratis | 6,275 | 633 | 6,908 |
| sppg | 4,977 | 1,534 | 6,511 |
| bgn | 3,331 | 1,233 | 4,564 |
| embege | 3,183 | 801 | 3,984 |
| badan gizi nasional | 3,568 | 29 | 3,597 |
| satuan pelayanan pemenuhan gizi | 1,495 | 0 | 1,495 |
| **All** | **28,081** | **7,633** | **35,714** |

## 11. Glossary

- **Post:** one X/Twitter post (tweet, reply or quote).
- **Keyword:** a search word or phrase (7 in total).
- **Cell:** one keyword in one month (154 cells).
- **Capped cell:** a cell where we asked for 300 posts and got 300, so more existed.
- **Batch 1 / `month_wide`:** posts from monthly searches.
- **Batch 2 / `slot_sample`:** posts from short 6-hour windows.
- **Slot:** one 6-hour window.
- **Duplicate:** a post whose text is identical to another after removing links.
- **Topic document:** one distinct text, the unit used for topic modelling.
- **Pseudonymised id (`user_hash`):** a one-way code replacing the author id; it cannot be reversed without the secret salt.
- **UTC:** world time; Indonesia (WIB) is UTC+7.

## 12. Literature found during the project (verify before citing)

These came up while looking for support for the keyword-discovery idea. **Journal indexing tiers were not verified.**
- Bozarth & Budak (2022), keyword-based data collection frameworks, EPJ Data Science 11:30.
- Kaban et al. (2025), BERT-based query expansion on Indonesian tweets, Eastern-European Journal of Enterprise Technologies.
- King, Lam & Roberts (2017), computer-assisted keyword expansion, American Journal of Political Science.
- Reimers & Gurevych (2020), multilingual sentence embeddings, EMNLP.
- Koto et al. (2021), IndoBERTweet, EMNLP.
- A 2025 paper combining IndoBERT with BERTopic (International Journal of Artificial Intelligence), cited as an example of the planned topic-modelling approach.
