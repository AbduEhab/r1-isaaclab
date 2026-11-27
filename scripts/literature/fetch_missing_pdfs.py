"""Fetch PDFs for the literature corpus that have no OpenAlex open-access URL.

Two fallbacks:
  1. Semantic Scholar batch API -> openAccessPdf URL (works for many arXiv/author copies),
  2. arXiv API title search -> first matching paper's PDF.

Run after build_literature.py:
    C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\literature\\fetch_missing_pdfs.py
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = r"C:\per\thesis\literature"
PDFS = os.path.join(ROOT, "pdfs")


def slug(s: str, n: int = 45) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:n]


def filename(row) -> str:
    return f"{row['id']}_{row['year']}_{slug(row['title'])}.pdf"


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (literature fallback)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def post_json(url, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0 (literature fallback)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def s2_batch(rows):
    """Ask Semantic Scholar for open-access PDF links in batches of 100 DOIs."""
    found = {}
    with_doi = [r for r in rows if r.get("doi")]
    for i in range(0, len(with_doi), 100):
        chunk = with_doi[i:i + 100]
        ids = [f"DOI:{r['doi']}" for r in chunk]
        for attempt in range(4):
            try:
                res = post_json("https://api.semanticscholar.org/graph/v1/paper/batch"
                                "?fields=title,openAccessPdf,externalIds", {"ids": ids})
                for row, item in zip(chunk, res):
                    if item and item.get("openAccessPdf", {}).get("url"):
                        found[row["id"]] = item["openAccessPdf"]["url"]
                print(f"[s2] batch {i // 100}: {len(found)} urls so far", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[s2] batch failed ({exc}); retry {attempt + 1}", flush=True)
                time.sleep(3.0 * (attempt + 1))
        time.sleep(1.0)
    return found


ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_lookup(row):
    title = re.sub(r"[^A-Za-z0-9 ]", " ", row["title"])[:120]
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"
    for attempt in range(2):
        try:
            xml = get(url, timeout=30).decode("utf-8", "ignore")
            root = ET.fromstring(xml)
            for entry in root.findall("a:entry", ARXIV_NS):
                pdf = entry.find("a:link[@title='pdf']", ARXIV_NS)
                if pdf is not None:
                    return pdf.get("href")
            return ""
        except Exception:  # noqa: BLE001
            time.sleep(2.0)
    return ""


def download(url, row):
    path = os.path.join(PDFS, filename(row))
    if os.path.exists(path) and os.path.getsize(path) > 20_000:
        return True
    try:
        data = get(url, timeout=60)
        if data.startswith(b"%PDF") and len(data) > 20_000:
            with open(path, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! download failed {row['id']}: {exc}", flush=True)
    return False


def main() -> int:
    os.makedirs(PDFS, exist_ok=True)
    rows = json.load(open(os.path.join(ROOT, "papers.json"), encoding="utf-8"))
    missing = [r for r in rows if not r.get("pdf")]
    print(f"[fallback] {len(missing)} papers without a PDF")

    found = s2_batch(missing)
    print(f"[fallback] semantic scholar gave {len(found)} urls")

    def try_row(r):
        if r["id"] in found:
            return r["id"], found[r["id"]]
        url = arxiv_lookup(r)
        return r["id"], url

    urls = dict(found)
    todo = [r for r in missing if r["id"] not in urls]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for wid, url in pool.map(try_row, todo):
            if url:
                urls[wid] = url
    print(f"[fallback] {len(urls)} candidate urls total")

    by_id = {r["id"]: r for r in rows}
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(download, url, by_id[wid]): wid for wid, url in urls.items()}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            wid = futs[fut]
            if fut.result():
                by_id[wid]["pdf"] = filename(by_id[wid])
                ok += 1
            if i % 20 == 0:
                print(f"[fallback] {i}/{len(futs)} attempted, {ok} downloaded", flush=True)

    with open(os.path.join(ROOT, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(ROOT, "papers.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    still = [r for r in rows if not r.get("pdf")]
    with open(os.path.join(ROOT, "missing_pdfs.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["nr", "year", "title", "doi", "venue"])
        for r in still:
            wr.writerow([r["nr"], r["year"], r["title"], r["doi"], r["venue"]])
    print(f"[fallback] done: +{ok} PDFs, {len(still)} still without a PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
