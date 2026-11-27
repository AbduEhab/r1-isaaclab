"""Hunt open-access copies for papers still missing a PDF.

Sources per missing paper:
  1. Unpaywall (repository / publisher OA locations, url_for_pdf),
  2. OpenAlex (all locations, not only best_oa_location),
  3. Semantic Scholar title search (last resort, rate-limited).

Run after fetch_missing_pdfs.py:
    C:\\per\\thesis\\.venv\\Scripts\\python.exe scripts\\literature\\hunt_oa_pdfs.py
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import time
import urllib.parse
import urllib.request

ROOT = r"C:\per\thesis\literature"
PDFS = os.path.join(ROOT, "pdfs")
EMAIL = "moabde1@emea.corpdir.net"


def slug(s, n=45):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:n]


def name_for(row):
    return f"{row['id']}_{row['year']}_{slug(row['title'])}.pdf"


def get(url, timeout=35):
    req = urllib.request.Request(url, headers={"User-Agent": f"Mozilla/5.0 (research; {EMAIL})"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def unpaywall_urls(doi):
    if not doi:
        return []
    try:
        d = json.loads(get(f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}").decode("utf-8", "ignore"))
    except Exception:  # noqa: BLE001
        return []
    urls = []
    for loc in ([d.get("best_oa_location")] + (d.get("oa_locations") or [])):
        if not loc:
            continue
        for key in ("url_for_pdf", "url_for_landing_page", "url"):
            u = loc.get(key)
            if u and u not in urls:
                urls.append(u)
    return urls


def openalex_urls(doi):
    if not doi:
        return []
    try:
        d = json.loads(get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}").decode("utf-8", "ignore"))
    except Exception:  # noqa: BLE001
        return []
    urls = []
    for loc in (d.get("locations") or []):
        for key in ("pdf_url", "landing_page_url"):
            u = (loc or {}).get(key)
            if u and u not in urls:
                urls.append(u)
    return urls


def s2_urls(title):
    q = urllib.parse.quote(title[:150])
    try:
        d = json.loads(get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
                           "&fields=title,year,openAccessPdf&limit=3").decode("utf-8", "ignore"))
    except Exception:  # noqa: BLE001
        return []
    urls = []
    want = re.sub(r"[^a-z0-9]", "", title.lower())[:60]
    for p in d.get("data", []):
        got = re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())[:60]
        if want[:40] and want[:40] in got:
            u = (p.get("openAccessPdf") or {}).get("url")
            if u:
                urls.append(u)
    return urls


def try_download(row, candidates):
    path = os.path.join(PDFS, name_for(row))
    if os.path.exists(path) and os.path.getsize(path) > 20_000:
        return path
    for url in candidates:
        if not url.lower().endswith(".pdf") and "pdf" not in url.lower():
            continue
        try:
            data = get(url, timeout=60)
            if data.startswith(b"%PDF") and len(data) > 20_000:
                with open(path, "wb") as f:
                    f.write(data)
                return path
        except Exception:  # noqa: BLE001
            continue
    return ""


def main() -> int:
    os.makedirs(PDFS, exist_ok=True)
    rows = json.load(open(os.path.join(ROOT, "papers.json"), encoding="utf-8"))
    missing = [r for r in rows if not r.get("pdf")]
    print(f"[hunt] {len(missing)} missing")

    def work(row):
        cands = unpaywall_urls(row.get("doi", ""))
        time.sleep(0.15)
        cands += [u for u in openalex_urls(row.get("doi", "")) if u not in cands]
        time.sleep(0.05)
        if not cands:
            cands += s2_urls(row["title"])
            time.sleep(1.2)
        path = try_download(row, cands)
        return row["id"], path, cands[:3]

    found = 0
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(work, r) for r in missing]
        by_id = {r["id"]: r for r in rows}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            wid, path, cands = fut.result()
            if path:
                by_id[wid]["pdf"] = os.path.basename(path)
                found += 1
            if i % 20 == 0:
                print(f"[hunt] {i}/{len(futs)} checked, {found} recovered", flush=True)

    with open(os.path.join(ROOT, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    import csv
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
    print(f"[hunt] recovered {found} PDFs, {len(still)} still missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
