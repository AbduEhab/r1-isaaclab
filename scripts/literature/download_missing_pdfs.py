"""Portable downloader for the 83 paywalled papers in the literature corpus.

Run this on a machine with valid University of Stuttgart access (campus network, TIK VPN, or UB proxy).
It resolves each DOI, reads the publisher's PDF link (citation_pdf_url / download link), and saves the
PDF under the same name used by the rest of the corpus.

Only stdlib + `requests` are required:  pip install requests

Usage (Windows):
    python download_missing_pdfs.py --list missing_papers.csv --out pdfs
    python download_missing_pdfs.py --list missing_papers.csv --out pdfs --proxy http://proxy.ub.uni-stuttgart.de:3128
    python download_missing_pdfs.py --list missing_papers.csv --out pdfs --check        # test access only
    python download_missing_pdfs.py --list missing_papers.csv --out pdfs --limit 5

Output:
    PDFs in --out (skip existing files, resumable)
    download_results.csv with per-paper status
    failed.txt with the DOIs/landing pages that still need to be opened manually
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}

PDF_KEYS = ("application/pdf", "application/x-pdf", "application/octet-stream")
META_PATTERNS = [
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
    r'<meta[^>]+property=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
]
PDF_LINK_PATTERNS = [
    r'href=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']',
    r'href=["\']([^"\']*(?:pdf|download)[^"\']*)["\']',
]


def get_session(proxy: str | None, timeout: int) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    s.request_timeout = timeout  # type: ignore[attr-defined]
    return s


def is_pdf(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if any(k in ctype for k in PDF_KEYS):
        return True
    head = resp.content[:5] if resp.content else b""
    return head == b"%PDF"


def save_pdf(content: bytes, out_dir: str, filename: str) -> str:
    if not content.startswith(b"%PDF") or len(content) < 20_000:
        return ""
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def fetch(session, url, timeout, delay):
    time.sleep(delay)
    return session.get(url, timeout=timeout, allow_redirects=True)


def extract_links(html: str, base_url: str) -> list[str]:
    links = []
    for pat in META_PATTERNS:
        for m in re.finditer(pat, html, re.I):
            links.append(m.group(1))
    for pat in PDF_LINK_PATTERNS:
        for m in re.finditer(pat, html, re.I):
            u = m.group(1)
            if u.startswith("/"):
                parts = requests.utils.urlparse(base_url)
                u = f"{parts.scheme}://{parts.netloc}{u}"
            elif not u.startswith("http"):
                continue
            links.append(u)
    # dedupe, prefer pdf-looking links
    seen, out = set(), []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    out.sort(key=lambda u: (".pdf" not in u.lower(), "citation_pdf" not in u.lower()))
    return out[:12]


def crossref_candidates(session, doi, timeout, delay):
    """Ask Crossref for publisher-declared PDF links (works for Springer, Wiley, Elsevier, ...)."""
    if not doi:
        return []
    import urllib.parse
    try:
        r = fetch(session, f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", timeout, delay)
        msg = r.json().get("message", {})
    except Exception:  # noqa: BLE001
        return []
    urls = []
    for link in (msg.get("link") or []):
        if "pdf" in (link.get("content-type") or "").lower() and link.get("URL"):
            urls.append(link["URL"])
    prim = ((msg.get("resource") or {}).get("primary") or {}).get("URL")
    if prim:
        urls.append(prim)
    return urls


def try_download(session, row, out_dir, timeout, delay, results):
    filename = row["filename"]
    path = os.path.join(out_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 20_000:
        results.append((row["id"], "exists", "", path))
        return True

    candidates = [u.strip() for u in (row.get("candidates") or "").split(";") if u.strip()]
    doi = (row.get("doi") or "").strip()
    for u in crossref_candidates(session, doi, timeout, delay):
        if u not in candidates:
            candidates.append(u)
    doi_url = (row.get("doi_url") or "").strip()
    if doi_url:
        candidates.append(doi_url)
    seen_cand = set()
    for cand in candidates:
        if cand in seen_cand:
            continue
        seen_cand.add(cand)
        try:
            r = fetch(session, cand, timeout, delay)
        except Exception as exc:  # noqa: BLE001
            results.append((row["id"], "error", cand, str(exc)[:120]))
            continue
        if r.status_code >= 400:
            results.append((row["id"], f"http_{r.status_code}", cand, r.url))
            continue
        if is_pdf(r):
            ok = save_pdf(r.content, out_dir, filename)
            results.append((row["id"], "ok" if ok else "bad_pdf", cand, r.url))
            return bool(ok)
        # IEEE: build the PDF URL from the arnumber if the landing page exposes it
        for m in re.finditer(r"arnumber[=:/](\d{4,})", r.text + " " + r.url):
            arn = m.group(1)
            for ieee in (f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arn}",
                         f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arn}"):
                try:
                    r3 = fetch(session, ieee, timeout, delay)
                except Exception:  # noqa: BLE001
                    continue
                if r3.status_code < 400 and is_pdf(r3):
                    ok = save_pdf(r3.content, out_dir, filename)
                    results.append((row["id"], "ok" if ok else "bad_pdf", ieee, r3.url))
                    return bool(ok)
            break
        # HTML: look for the publisher's PDF link and follow it once
        for link in extract_links(r.text, r.url):
            try:
                r2 = fetch(session, link, timeout, delay)
            except Exception:  # noqa: BLE001
                continue
            if r2.status_code < 400 and is_pdf(r2):
                ok = save_pdf(r2.content, out_dir, filename)
                results.append((row["id"], "ok" if ok else "bad_pdf", link, r2.url))
                return bool(ok)
        results.append((row["id"], "no_pdf_link", cand, r.url))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="missing_papers.csv")
    ap.add_argument("--out", default="pdfs")
    ap.add_argument("--proxy", default="", help="e.g. http://proxy.ub.uni-stuttgart.de:3128")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only probe access, do not download")
    ap.add_argument("--insecure", action="store_true",
                    help="disable TLS verification (needed behind a corporate TLS-inspecting proxy)")
    ap.add_argument("--results", default="download_results.csv")
    args = ap.parse_args()

    if not os.path.exists(args.list):
        sys.exit(f"list not found: {args.list}")
    os.makedirs(args.out, exist_ok=True)
    rows = list(csv.DictReader(open(args.list, encoding="utf-8-sig")))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[dl] {len(rows)} papers, out={args.out}, proxy={args.proxy or 'none'}, delay={args.delay}s")

    if args.insecure:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    session = get_session(args.proxy or None, args.timeout)
    session.verify = not args.insecure
    results = []
    ok = 0
    for i, row in enumerate(rows, 1):
        if args.check:
            url = row.get("doi_url") or ""
            try:
                r = fetch(session, url, args.timeout, args.delay) if url else None
                status = f"http_{r.status_code}" if r is not None else "no_url"
                print(f"[check] {i}/{len(rows)} {row['id']} {status} {r.url if r else ''}")
            except Exception as exc:  # noqa: BLE001
                print(f"[check] {i}/{len(rows)} {row['id']} error {exc}")
            continue
        success = try_download(session, row, args.out, args.timeout, args.delay, results)
        ok += int(success)
        status = results[-1][1] if results else "?"
        print(f"[dl] {i}/{len(rows)} {row['id']} {status} | {row['title'][:60]}", flush=True)

    if not args.check:
        with open(args.results, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["id", "status", "url", "detail"])
            wr.writerows(results)
        failed = [r for r in results if r[1] not in ("ok", "exists")]
        with open("failed.txt", "w", encoding="utf-8") as f:
            for r in failed:
                f.write(r[0] + "\n")
        # a clickable manual list for the leftovers
        by_id = {r["id"]: r for r in rows}
        with open("manual_todo.txt", "w", encoding="utf-8") as f:
            for r in failed:
                row = by_id.get(r[0], {})
                f.write(f"{row.get('year','')} | {row.get('title','')}\n    {row.get('doi_url','')} {row.get('candidates','')}\n\n")
        print(f"\n[dl] done: {ok}/{len(rows)} downloaded; see download_results.csv and failed.txt")
        print("[dl] failed papers can be opened manually via https://doi.org/<doi> while on the university network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
