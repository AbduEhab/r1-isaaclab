# Download the remaining paywalled PDFs

The corpus has **300 papers; 217 PDFs** are already downloaded. The remaining **83** are listed in
`missing_papers.csv` (columns: DOI, `doi_url`, publisher-direct candidate URLs, target filename).

Run this on a machine with valid Uni Stuttgart access (campus network, TIK VPN, or the UB proxy):

```powershell
pip install requests
cd <folder with download_missing_pdfs.py and missing_papers.csv>
python download_missing_pdfs.py --list missing_papers.csv --out pdfs
```

Useful flags:

| flag | use |
|---|---|
| `--check` | only probe which DOIs resolve from this network (no downloads) |
| `--proxy http://proxy.ub.uni-stuttgart.de:3128` | route through the UB proxy |
| `--insecure` | skip TLS verification (needed behind a corporate TLS-inspecting proxy) |
| `--limit 5` | test on the first 5 papers |
| `--delay 2` | slower, politer pacing |

Outputs: PDFs into `--out` (same filenames as the corpus, resumable), `download_results.csv`,
`failed.txt`, and `manual_todo.txt` (clickable DOI links for whatever still fails).

Notes:
- IEEE (16 ICRA + 5 IROS + T-RO/RAM) sometimes blocks scripted downloads even with access; those land in
  `manual_todo.txt` — open the DOI in a browser on the campus network and click "PDF".
- The script uses DOI + Crossref publisher links + each publisher's direct PDF pattern; no credentials are
  stored anywhere.
- After the run, copy the downloaded PDFs into `literature/pdfs/`; `papers.json`/`papers.csv` can be
  refreshed by re-running `scripts/literature/download_pdfs.py` (it skips existing files).
