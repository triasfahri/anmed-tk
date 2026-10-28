# Preprocessing Twitter/X MBG

Pendekatan, keputusan desain, dan alasannya dijelaskan di [`PREPROCESSING_APPROACH.md`](PREPROCESSING_APPROACH.md). File ini berisi petunjuk operasional.

Pipeline ini memproses `raw_data_MBG.jsonl` (35.714 posting berbahasa Indonesia dalam JSON Lines) dengan Python dan Pandas. Desainnya memisahkan kebutuhan RQ1/topic modeling dari RQ2/engagement dan RQ3/temporal agar deduplikasi tidak mengubah intensitas percakapan atau statistik engagement.

Input merupakan gabungan dua batch: 28.081 posting dari query bulanan (`final_posts.jsonl`, `pull_batch=month_wide`) dan 7.633 posting dari query slot enam jam yang dirotasi (`extra_posts.jsonl`, `pull_batch=slot_sample`). Batch pertama condong ke posting populer; batch kedua menambahkan sampel dari slot waktu di dalam sel keyword-bulan yang capped.

## Output utama

```text
output/
├── posts_preprocessed.jsonl      # seluruh posting valid dalam JSON Lines; isi sama dengan posts_rq2_rq3.csv
├── posts_rq2_rq3.csv             # seluruh posting valid; tanpa deduplikasi teks
├── topic_documents_rq1.csv       # dokumen unik; URL diabaikan saat deduplikasi
├── manual_quality_audit.csv      # 150 sampel untuk penilaian manual
├── preprocessing_report.json     # retention dan konfigurasi pipeline
└── topic_stopwords.txt           # stopword CountVectorizer BERTopic
```

`collection_log.csv` di samping script mencatat jendela koleksi per keyword dan bulan UTC (`keyword,month,fetched,capped`), dengan batas 300 posting per jendela. Kolom `capped` berisi teks `True` atau `False`.

Validator menghasilkan `output/validation_report.json` berisi hasil 22 pemeriksaan otomatis. Output lama yang digantikan disimpan di `output/_superseded/`. Sebelum merge pertama, input lama dipindahkan ke `raw_data_MBG_batch1_month_wide.jsonl`, salinan batch 1 yang tetap utuh; arsip yang sudah ada tidak ditimpa. `.gitignore` mengecualikan `output/`, `__pycache__/`, `.DS_Store`, `.venv/`, dan `raw_data_MBG*.jsonl` dari Git.

## Instalasi dan menjalankan pipeline

```bash
cd path/to/anmed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python merge_raw.py
python preprocess_twitter.py
python validate_pipeline.py
```

`merge_raw.py` mencari `final_posts.jsonl` dan `extra_posts.jsonl` di `../data/` atau `../tugas-kelompok/data/`, relatif terhadap script. Gunakan `--old PATH`, `--new PATH`, dan `--out PATH` jika lokasi berbeda; output default adalah `raw_data_MBG.jsonl` di samping script. Merge selalu membaca kedua sumber, mempertahankan urutan batch 1 lalu batch 2, dan berhenti sebelum menulis atau memindahkan file jika ada ID ganda. Menjalankannya ulang menghasilkan output identik dan tidak mengubah arsip batch 1.

Untuk rerun dengan lingkungan yang dipakai pada validasi terakhir (pandas 2.2.3, emoji 2.15.0):

```bash
PYTHON=/private/tmp/claude-501/-Users-triasfahri-Documents-Sem7-Anmedsos/36d5c523-d519-417c-8a64-42b538eb42cc/scratchpad/pinned_env/bin/python
"$PYTHON" merge_raw.py
"$PYTHON" preprocess_twitter.py
"$PYTHON" validate_pipeline.py
```

Paket `emoji==2.15.0` adalah dependency wajib. Script berhenti dengan `ModuleNotFoundError` jika dependency belum dipasang; tidak ada fallback, sehingga emoji gabungan dan skin tone diproses konsisten antarlingkungan.

Opsi:

```text
--input PATH                  input JSONL/JSON/CSV/Parquet
--collection-log PATH         log jendela koleksi; default collection_log.csv
--posts-output PATH           seluruh posting valid individual RQ2/RQ3
--topic-output PATH           output dokumen unik RQ1
--audit-output PATH           lembar audit manual
--report PATH                 retention/configuration report
--stopwords-output PATH       stopword topik; default output/topic_stopwords.txt
--json-output PATH            seluruh posting valid dalam JSON Lines; default output/posts_preprocessed.jsonl
--audit-size 150              jumlah sampel audit
--seed 42                     seed audit reproducible
```

Input default `raw_data_MBG.jsonl`, log default `collection_log.csv`, dan seluruh output default berada relatif terhadap folder script, sehingga tidak bergantung pada working directory. Jika log tidak ditemukan, pipeline tetap berjalan dengan satu baris peringatan dan seluruh `window_capped` missing.

## Tahapan preprocessing

### 1. Loading dan validasi schema

- Mendukung JSON Lines, JSON array, CSV, dan Parquet.
- Memastikan `id`, `text`, dan `created_at` tersedia.
- ID dipertahankan sebagai string agar digit panjang tidak kehilangan presisi.
- Engagement dikonversi menjadi numerik.
- Tanggal dikonversi menjadi UTC, lalu baris diurutkan stabil dari tanggal paling awal dengan tanggal invalid di akhir; tanggal invalid ditandai sebelum dikeluarkan dari output analisis.

### 2. Populasi analisis

Semua analisis RQ1, RQ2, dan RQ3 memakai populasi seluruh posting valid: teks tidak kosong, tanggal valid, dan bukan duplicate ID. Filter relevansi dihapus karena recall pada sampel berlabel manusia sekitar 0,72 dan fokus penelitian adalah topic modelling. Kolom `judge_score` dan `on_topic` dibuang saat loading, tidak disertakan pada output apa pun, dan tetap hanya berada di input mentah; pipeline tetap berjalan jika kedua kolom tidak tersedia.

### 3. Data quality dan retention

Script menandai:

- `is_missing_text`;
- `is_invalid_date`;
- `is_duplicate_id`;
- `is_duplicate_text`;
- `duplicate_text_group` sebagai hash stabil untuk teks minimal tanpa URL, dengan whitespace dirapikan dan huruf kecil; jika hasilnya kosong, gunakan teks minimal huruf kecil.

`preprocessing_report.json` mencatat jumlah pada setiap tahap: input, missing text, tanggal invalid, duplicate ID/text, output posting individual, dan hasil deduplikasi RQ1. Hitungan informasional `window_capped_true_rows`, `window_capped_false_rows`, `window_capped_unknown_rows`, `pull_batch_month_wide_rows`, dan `pull_batch_slot_sample_rows` dihitung pada posting valid RQ2/RQ3; path log dicatat sebagai `configuration.collection_log`.

Hasil dataset saat ini:

- input dan posting valid RQ2/RQ3: 35.714; `posts_preprocessed.jsonl` memuat 35.714 record yang sama;
- missing text, tanggal invalid, dan duplicate ID: 0;
- duplicate berdasarkan teks minimal tanpa URL dan tanpa membedakan kapitalisasi: 559 baris setelah kemunculan pertama (381 kelompok mencakup 940 posting);
- dokumen unik untuk RQ1: 35.155;
- `window_capped=True`: 30.525 posting; `False`: 5.189; unknown: 0;
- `pull_batch=month_wide`: 28.081 posting; `slot_sample`: 7.633;
- sampel audit dari `posts_rq2_rq3.csv`: 150 (`seed=42`).

### 4. Unicode, HTML, URL, mention, dan hashtag

- HTML entity seperti `&amp;` didecode.
- Unicode dinormalisasi dengan NFKC.
- Control character dan whitespace dinormalisasi.
- Pada `text_normalized`, URL menjadi `[URL]` dan mention menjadi `[USER]`, lalu menjadi `[url]`/`[user]` setelah lowercase. Kedua placeholder dibuang dari `text_topic`.
- Pada `text_embed`, URL dan deretan mention di awal dibuang; emoji serta mention di tengah kalimat dipertahankan.
- Hashtag camelCase/snake_case dan batas akronim disegmentasi: `#KendalaDanaSPPGSolo` menjadi `kendala dana sppg solo`. Kamus `HASHTAG_FIXES` menangani bentuk menyatu seperti `#awasimbg` menjadi `awasi mbg`.

Segmentasi hashtag berbasis aturan tidak selalu sempurna, sehingga hasilnya masuk ke audit manual.

### 5. Emoji

Emoji dikonversi menjadi token yang dapat dibaca vectorizer, misalnya:

```text
MBG basi 🤢 -> mbg basi emoji_nauseated_face
```

Modifier skin tone U+1F3FB–U+1F3FF dibuang sebelum demojize sehingga `👍🏽` dan `👍` menghasilkan token yang sama; urutan ZWJ tetap dipertahankan. `text_minimal` dan `text_embed` tetap menyimpan emoji asli. Kolom `has_emoji` memudahkan analisis dan sampling. Strategi ini menjaga sinyal keluhan atau emosi pada representasi topik.

### 6. Slang dan repeated characters

Kamus slang mencakup bentuk seperti `gak`, `gmn`, `knp`, `emang`, `udh`, dan `dpt`. `dr` dipertahankan agar singkatan dokter tidak berubah menjadi `dari`. Negasi seperti `tidak`, `bukan`, `jangan`, dan `belum` dipertahankan.

Repetisi hanya diruntuhkan jika panjangnya tiga atau lebih:

| Input | Output |
|---|---|
| `telattt` | `telat` |
| `baguuuus` | `bagus` |
| `gakkk` | `tidak` setelah slang mapping |
| `lamaaaa` | `lama` |
| `massa` | `massa` |
| `Allah` | `allah` setelah lowercase; huruf ganda tetap utuh |

Kamus harus diperluas berdasarkan temuan corpus dan didokumentasikan sebagai keputusan penelitian.

## Empat representasi teks dan BERTopic

| Kolom | Isi | Penggunaan |
|---|---|---|
| `text` | teks sumber asli | audit/reproduksibilitas |
| `text_minimal` | Unicode, HTML, control character, whitespace | kandidat embedding pembanding |
| `text_embed` | teks minimal tanpa URL dan deretan mention pembuka; emoji asli dipertahankan | baseline embedding yang disarankan |
| `text_normalized` | placeholder `[URL]`/`[USER]` dalam lowercase, hashtag, emoji token, repetisi, slang | kandidat embedding alternatif |
| `text_topic` | token normalized tanpa placeholder + stopword konservatif; kata query tetap ada | c-TF-IDF dan representasi/label topik |

Kolom tambahan pada posting dan dokumen topik:

| Kolom | Isi |
|---|---|
| `dup_group_size` | jumlah seluruh posting valid dalam `duplicate_text_group`, dihitung sebelum deduplikasi RQ1 |
| `source_keyword` | nilai sumber yang sudah tersedia dipertahankan; hanya nilai missing diisi dari bagian `query` sebelum ` since:`, tanpa tanda kutip ganda pembuka/penutup |
| `source_month` | nilai sumber `YYYY-MM` yang sudah tersedia dipertahankan; hanya nilai missing diisi dari tanggal setelah `since:` pada `query` |
| `pull_batch` | `month_wide` untuk batch 1, `slot_sample` untuk batch 2; kolom/nilai yang missing diisi `month_wide` |
| `slot_since`, `slot_until` | batas waktu slot ISO UTC dari batch 2, diteruskan tanpa perubahan; missing untuk batch 1 |
| `window_capped` | nullable boolean dari `collection_log.csv` menurut pasangan `source_keyword` dan `source_month`; keyword dicocokkan persis, missing jika jendela tidak tercatat |
| `uses_embege` | boolean untuk kata utuh `embege`, tanpa membedakan kapitalisasi; huruf ASCII, angka, dan underscore dianggap bagian kata |

Query batch 2 memakai `since_time:`/`until_time:` dan pengecualian keyword; `source_keyword` dan `source_month` yang diberikan kolektor tetap dipakai. `window_capped` dicari berdasarkan sel `(source_keyword, source_month)` di log koleksi, sehingga seluruh baris `slot_sample` bernilai `True`: slot tersebut disampel di dalam sel yang capped, bukan karena query slotnya sendiri mencapai batas.

Pada dokumen topik RQ1, metadata sumber, batch, slot, dan `window_capped` menggambarkan posting layak paling awal yang mewakili kelompok duplikat, bukan seluruh anggota kelompok. Metadata ini tidak disertakan pada lembar audit manual.

Untuk privasi, `bio` hanya berada di input mentah dan dibuang dari seluruh output posting, dokumen topik, serta lembar audit.

Prinsip penting: **jangan memakai `text_topic` untuk seluruh tahap BERTopic**. Stopword removal dapat membuang konteks yang dibutuhkan Sentence Transformer.

Alur eksperimen yang disarankan:

```text
Embedding baseline    : topic_documents_rq1["text_embed"]
Embedding alternatif  : topic_documents_rq1["text_minimal"] atau ["text_normalized"]
Representasi c-TF-IDF : update_topics dengan text_topic dan topic_stopwords.txt
Label/interpretasi    : audit text_topic bersama teks asli
```

`topic_stopwords.txt` berisi gabungan terurut `STOPWORDS_ID`, `QUERY_STOPWORDS`, dan `FILLER_STOPWORDS`, satu kata per baris. Stopword query/filler ini dipakai hanya oleh CountVectorizer, tanpa menghapus kata query dari `text_topic`. Kata `gizi` dan `program` tetap tersedia.

Contoh berikut **belum diuji**; cocokkan dengan versi BERTopic yang terpasang dan model embedding yang dipilih. Setelah fit pada `text_embed`, perbarui representasi dengan `text_topic` dalam urutan dokumen yang sama melalui [`update_topics`](https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic.BERTopic.update_topics).

```python
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

documents = pd.read_csv("output/topic_documents_rq1.csv", dtype={"id": "string"})
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(documents["text_embed"].tolist())

vectorizer_model = CountVectorizer(
    stop_words=Path("output/topic_stopwords.txt").read_text(encoding="utf-8").splitlines(),
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 2),
)
topic_model.update_topics(
    documents["text_topic"].fillna("").tolist(),
    topics=topics,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
)
```

## Aturan deduplikasi per research question

### RQ1 - BERTopic

`topic_documents_rq1.csv`:

- memakai seluruh posting valid tanpa filter `on_topic` atau `judge_score`;
- membuang missing text, tanggal invalid, dan duplicate ID;
- memerlukan `text_normalized` dan `text_embed` yang tidak kosong;
- menyatukan `text_minimal` setelah URL dibuang, whitespace dirapikan, dan huruf dijadikan lowercase; jika kosong, kunci memakai teks minimal lowercase;
- mempertahankan posting layak bertanggal paling awal; jika tanggal sama, urutan input dipertahankan;
- memiliki `topic_document_id` dan `dup_group_size` yang menghitung seluruh posting valid dalam kelompok sebelum deduplikasi.

### RQ2 - Engagement dan RQ3 - Temporal

`posts_rq2_rq3.csv`:

- memakai seluruh posting valid, konsisten dengan populasi RQ1 sebelum deduplikasi teks;
- mempertahankan setiap posting valid beserta engagement dan waktu;
- tidak membuang duplicate text maupun posting dengan `text_embed` kosong;
- memiliki `duplicate_text_group`, `dup_group_size`, dan `topic_document_id`; ID dokumen boleh kosong hanya pada posting dengan `text_embed` kosong.

Setelah BERTopic menghasilkan assignment, gabungkan kembali berdasarkan `topic_document_id`:

```python
import pandas as pd

posts = pd.read_csv("output/posts_rq2_rq3.csv", dtype={"id": "string"})
topics = pd.read_csv("output/topic_documents_rq1.csv", dtype={"id": "string"})

# Contoh setelah kolom Topic ditambahkan ke topics oleh BERTopic.
topic_map = topics[["topic_document_id", "Topic"]]
posts_with_topics = posts.merge(topic_map, on="topic_document_id", how="left")
```

Setiap `topic_document_id` yang terisi pada posting memetakan kelompok teks tanpa URL ke satu dokumen RQ1. Penggabungan kiri tetap mempertahankan posting tanpa dokumen topik.

Untuk RQ3, gunakan proporsi topik per bulan, lalu opsional ulangi analisis dengan `window_capped == False` sebagai uji robustness. Jumlah posting pada jendela yang mencapai batas bukan ukuran volume percakapan bulanan. Analisis engagement dan analisis lain yang peka terhadap popularitas harus memakai `pull_batch` untuk membandingkan hasil per batch atau mengendalikan perbedaan desain koleksi: median likes `month_wide` adalah 19 secara keseluruhan dan 663 untuk `mbg`, dibanding 5 dan 11 pada `slot_sample`. Menggabungkan keduanya tanpa memperhitungkan batch mencampur desain sampling yang berbeda. `window_capped` saja tidak membedakan kedua batch karena seluruh `slot_sample` berada di sel yang capped.

## Manual quality assessment

`manual_quality_audit.csv` berisi 150 sampel reproducible (`seed=42`) dari `posts_rq2_rq3.csv` dengan raw dan keempat versi teks. Isi kolom:

- `meaning_preserved`: misalnya `ya`, `sebagian`, atau `tidak`;
- `review_status`: ubah dari `belum_dinilai` menjadi `selesai`;
- `reviewer_notes`: catat emoji, hashtag, slang, code-mixing, atau perubahan makna.

Audit idealnya dilakukan oleh minimal satu peneliti, dan subset dapat dinilai oleh peneliti kedua untuk memeriksa konsistensi.

## Testing

Jalankan:

```bash
python validate_pipeline.py
```

Validator memeriksa 22 area: syntax Python, merge identik saat diulang dan penolakan ID ganda tanpa mengubah file, jumlah raw sesuai retention, retention setiap tahap termasuk hitungan capped, kesegaran output, struktur 150 sampel, kasus transformasi, kasus deduplikasi, kasus pemetaan jendela dan log hilang, fixture metadata dua batch, teks wajib yang kosong, konsistensi populasi RQ1-RQ3, kelengkapan input BERTopic, penghapusan `bio`, `judge_score`, dan `on_topic`, teks embedding, ukuran kelompok duplikat, placeholder topik, stopword, metadata sumber termasuk format bulan, kesesuaian flag dengan log koleksi, kesetaraan JSONL dengan CSV, serta label/hitungan batch dan batas slot. Hitungan batch dibandingkan dengan raw dan report tanpa angka populasi tetap. Pemeriksaan kesegaran menjalankan pipeline dengan interpreter yang sama ke folder sementara, lalu membandingkan SHA-256 tiga CSV, JSONL, report, dan stopword; hanya path stopword sementara di report yang disamakan sebelum hashing. Jika berbeda, validator meminta pipeline dijalankan ulang.

Hasil terakhir: **22 pemeriksaan otomatis lulus, 0 gagal**. Mekanisme audit 150 baris (`manual_quality_audit.csv`) disediakan, tetapi review semantik sengaja dilewati sesuai keputusan tim, sehingga `review_status` tetap `belum_dinilai`; validator melaporkan `0/150` karena alasan tersebut. Evaluasi topik (coherence, diversity, interpretabilitas) dilakukan pada tahap pemodelan, di luar cakupan pipeline ini.

## Audit panjang teks dan code-mixing

Report mencatat maksimum dan persentil ke-95 jumlah token berbasis whitespace. Ini hanya pemeriksaan awal, bukan batas model.

Setelah model embedding dipilih:

1. Jalankan tokenizer resmi model terhadap `text_embed`, `text_minimal`, dan `text_normalized`.
2. Catat proporsi dokumen yang melewati `model_max_length`.
3. Pilih truncation atau chunking secara konsisten.
4. Jangan memangkas berdasarkan jumlah karakter.
5. Audit sampel code-mixing Indonesia-Inggris; pipeline sengaja tidak menerjemahkannya.

## Batasan

- Segmentasi hashtag tidak memakai model word segmentation.
- Kamus slang belum lengkap dan dapat mengubah istilah domain jika ditambah tanpa audit.
- Deduplikasi hanya menyamakan teks minimal tanpa URL, whitespace berlebih, dan perbedaan kapitalisasi; belum near-duplicate.
- Hasil emoji bergantung pada versi dependency; karena itu versi `emoji` dikunci di `requirements.txt` dan dicatat dalam report.
- Stemming tidak digunakan karena berpotensi mengurangi interpretabilitas dan harus diuji sebagai variabel eksperimen terpisah.
