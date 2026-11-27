"""Download open-access PDFs for the literature corpus and write the folder index.

Reads C:\\per\\thesis\\literature\\papers.json (produced by build_literature.py), downloads every paper that
has an open-access PDF (arXiv / publisher OA), names files by number-year-slug, and rewrites papers.csv,
papers.json and README.md with the local PDF path (or a `missing` note + DOI for paywalled work).

Run:  C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\literature\\download_pdfs.py
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import urllib.request

ROOT = r"C:\per\thesis\literature"
PDFS = os.path.join(ROOT, "pdfs")


def slug(s: str, n: int = 55) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:n]


def pdf_url(row: dict) -> str:
    doi = (row.get("doi") or "").lower()
    oa = row.get("oa_url") or ""
    m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", oa + " " + doi)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    if oa.lower().endswith(".pdf"):
        return oa
    return ""


def download(row: dict) -> tuple[int, str]:
    nr = row["nr"]
    url = pdf_url(row)
    name = f"{row['id']}_{row['year']}_{slug(row['title'], 45)}.pdf"
    path = os.path.join(PDFS, name)
    if os.path.exists(path) and os.path.getsize(path) > 20_000:
        return nr, name
    if not url:
        return nr, ""
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (literature downloader)"})
            with urllib.request.urlopen(req, timeout=50) as r, open(path, "wb") as f:
                data = r.read()
                if not data.startswith(b"%PDF") or len(data) < 20_000:
                    raise ValueError("not a pdf")
                f.write(data)
            return nr, name
        except Exception:  # noqa: BLE001
            if attempt == 1:
                if os.path.exists(path):
                    os.remove(path)
                return nr, ""
    return nr, ""


def main() -> int:
    os.makedirs(PDFS, exist_ok=True)
    rows = json.load(open(os.path.join(ROOT, "papers.json"), encoding="utf-8"))
    print(f"[pdf] {len(rows)} papers, {len(set(pdf_url(r) for r in rows if pdf_url(r)))} with an OA pdf url")
    by_nr = {r["nr"]: r for r in rows}
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, r) for r in rows]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            nr, name = fut.result()
            if name:
                by_nr[nr]["pdf"] = name
                ok += 1
            if i % 20 == 0:
                print(f"[pdf] {i}/{len(rows)} processed, {ok} downloaded", flush=True)

    rows = [by_nr[r["nr"]] for r in rows]
    with open(os.path.join(ROOT, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    import csv

    with open(os.path.join(ROOT, "papers.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    missing = [r for r in rows if not r.get("pdf")]
    with open(os.path.join(ROOT, "missing_pdfs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["nr", "year", "title", "doi", "venue"])
        for r in missing:
            wr.writerow([r["nr"], r["year"], r["title"], r["doi"], r["venue"]])

    # remove orphan PDFs (left over from earlier selections)
    keep = {r["pdf"] for r in rows if r.get("pdf")}
    removed = 0
    for f in os.listdir(PDFS):
        if f.endswith(".pdf") and f not in keep:
            os.remove(os.path.join(PDFS, f))
            removed += 1
    print(f"[pdf] cleaned {removed} orphan files")

    # README index
    from collections import Counter

    themes = Counter(r["theme"] for r in rows)
    years = Counter(r["year"] for r in rows)
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Literature corpus (citation-chained)\n\n")
        f.write(f"{len(rows)} papers, {ok} open-access PDFs downloaded, {len(missing)} without an OA PDF.\n\n")
        f.write("Built with `scripts/literature/build_literature.py`: ~60 curated seeds (classic robotics/SLAM to "
                "2026 frontier) -> their references (ancestors), the papers citing them (children) and the papers "
                "citing the most cited children (children of children). Metadata from OpenAlex.\n\n")
        f.write("## Files\n\n")
        f.write("| file | content |\n|---|---|\n")
        f.write("| `papers.csv` / `papers.json` | one row per paper: theme, provenance (seed/ancestor/child/grandchild), "
                "year, venue, citations, DOI, OA url, local PDF |\n")
        f.write("| `bibtex.bib` | BibTeX for all papers |\n")
        f.write("| `by_theme/*.md` | reading lists per theme, sorted by citations |\n")
        f.write("| `pdfs/` | downloaded open-access PDFs (named `<openalex-id>_<year>_<title-slug>.pdf`) |\n")
        f.write("| `missing_pdfs.csv` | papers without an open-access PDF (DOI included) |\n")
        f.write("| `_cache/` | raw OpenAlex API responses (kept for reproducibility) |\n\n")
        f.write("## Papers per theme\n\n")
        for theme, n in themes.most_common():
            f.write(f"- {theme}: {n}\n")
        f.write("\n## By year\n\n")
        for y in sorted(years):
            f.write(f"- {y}: {years[y]}\n")
        f.write("\n## Notes\n\n")
        f.write("- Paywalled papers are listed with DOI but no PDF; the metadata and BibTeX are still complete.\n")
        f.write("- Some 2026 preprints are indexed without citation counts yet (they are included as frontier seeds/"
                "descendants).\n")
    print(f"[pdf] done: {ok} PDFs, {len(missing)} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
