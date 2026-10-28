# MBG posts on X: collection and preprocessing

Code and documentation for collecting and preparing public X/Twitter posts about Indonesia's free-meals programme (MBG, Makan Bergizi Gratis). Course project (Anmedsos).

**Start here**
- [docs/APPROACH.md](docs/APPROACH.md): what we did and **why**, the caveats, and the numbers for the report.
- [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md): what each script does, how to run it, data files and columns.
- [preprocessing/PREPROCESSING_APPROACH.md](preprocessing/PREPROCESSING_APPROACH.md): the cleaning rules in detail.

**Layout**
- `collect.py`, `extra_pull.py`, `run_extra_pull.py`: collection (two batches).
- `preprocessing/`: merge, cleaning pipeline and validator.
- `test_*.py`: offline tests (no network, no credit).

**The data is not in this repository.** The repository is public and the raw files contain profile bios, so `data/`, the raw files and the generated outputs are gitignored. Ask the team for the data files and keep them out of git.

Secrets (`APIFY_TOKEN`, `USER_HASH_SALT`) live in environment variables, never in the repository.
