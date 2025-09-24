#!/usr/bin/env python3
"""
ArXiv PDF crawler/downloader.

Notes and ethics:
- Please be respectful to arXiv. This script uses the official arXiv API
  (export.arxiv.org/api) to discover records and downloads PDFs from arXiv
  with conservative rate limits and retries.
- For very large-scale downloads, prefer arXiv bulk access (e.g., Kaggle
  mirrors or institutional mirrors) instead of scraping. arXiv has
  practical and ethical limits; do not overload their infrastructure.

Features:
- Query arXiv via Atom API with pagination
- Download PDFs with concurrency, retry, backoff, and timeouts
- Resume via manifest JSONL file (records success/failure + file size)
- Optional limits: max records, max total bytes
- Stable output layout: one file per ID, sanitized

Examples:
  python data_crawler.py \
    --query "cat:cs.CL" \
    --from-date 2024-01-01 --until-date 2024-12-31 \
    --max-records 500 \
    --out-dir ./arxiv_pdfs \
    --concurrency 4

  python data_crawler.py --query "all:transformer" --max-total-bytes 500000000
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional, Tuple

import requests


ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
USER_AGENT = (
    "Mozilla/5.0 (compatible; arxiv-pdf-crawler/1.0; +https://arxiv.org)"
)


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    published: str


def build_search_query(query: str, from_date: Optional[str], until_date: Optional[str]) -> str:
    # arXiv API supports date ranges via submittedDate in queries: submittedDate:[YYYYMMDD TO YYYYMMDD]
    # Convert dates to YYYYMMDD if provided.
    date_clause = None
    if from_date or until_date:
        def compact(d: Optional[str]) -> Optional[str]:
            return d.replace("-", "") if d else None

        start = compact(from_date) or "00010101"
        end = compact(until_date) or "99991231"
        # Use spaces; requests will encode them. Avoid '+' which some servers don't treat as space inside ranges
        date_clause = f"submittedDate:[{start} TO {end}]"

    if query and date_clause:
        return f"({query}) AND {date_clause}"
    elif date_clause:
        return date_clause
    else:
        return query or "all:electron"  # default non-empty query


def parse_atom_entries(xml_text: str) -> List[ArxivEntry]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    entries: List[ArxivEntry] = []
    for e in root.findall("atom:entry", ns):
        id_url = e.findtext("atom:id", default="", namespaces=ns)
        title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
        published = e.findtext("atom:published", default="", namespaces=ns) or ""
        # id_url like: http://arxiv.org/abs/1234.5678v1
        arxiv_id = id_url.rsplit("/", 1)[-1]
        # Normalize to remove version suffix in filename while keeping version for URL if desired
        entries.append(ArxivEntry(arxiv_id=arxiv_id, title=title, published=published))
    return entries


def fetch_arxiv_batch(query: str, start: int, max_results: int, timeout: int = 30) -> Tuple[List[ArxivEntry], int]:
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(ARXIV_API, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    entries = parse_atom_entries(resp.text)

    # totalResults is optional; parse if present to inform paging
    try:
        root = ET.fromstring(resp.text)
        total_str = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        total = int(total_str.text) if total_str is not None and total_str.text else -1
    except Exception:
        total = -1
    return entries, total


def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "file"


def id_to_pdf_url(arxiv_id: str) -> str:
    # arxiv_id may contain version suffix e.g., 1234.5678v3
    return f"{ARXIV_PDF_BASE}{arxiv_id}.pdf"


def id_to_filename(arxiv_id: str) -> str:
    # Drop version for filename to avoid duplicates across versions
    base = re.sub(r"v\d+$", "", arxiv_id)
    return sanitize_filename(base) + ".pdf"


def save_manifest_line(manifest_path: str, record: dict) -> None:
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_pdf(arxiv_id: str, out_dir: str, session: requests.Session, timeout: int, max_retries: int, backoff: float, min_pdf_bytes: int = 0) -> Tuple[str, bool, int, Optional[str]]:
    url = id_to_pdf_url(arxiv_id)
    filename = id_to_filename(arxiv_id)
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, True, os.path.getsize(out_path), None

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, headers=headers, timeout=timeout, stream=True) as r:
                if r.status_code == 404:
                    return out_path, False, 0, "404 Not Found"
                r.raise_for_status()
                content_type = (r.headers.get("Content-Type") or "").lower()
                tmp_path = out_path + ".part"
                first_bytes = b""
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        if not first_bytes:
                            first_bytes = chunk[:8]
                        f.write(chunk)
                os.replace(tmp_path, out_path)
                size_bytes = os.path.getsize(out_path)
                # Validate PDF: content-type or magic header and minimum size
                is_pdf_header = first_bytes.startswith(b"%PDF-")
                is_pdf_ct = "application/pdf" in content_type
                if not (is_pdf_header or is_pdf_ct):
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                    return out_path, False, 0, f"Non-PDF response (Content-Type={content_type})"
                if min_pdf_bytes and size_bytes < min_pdf_bytes:
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                    return out_path, False, 0, f"File too small ({size_bytes} < {min_pdf_bytes})"
                return out_path, True, size_bytes, None
        except Exception as e:
            if attempt == max_retries:
                return out_path, False, 0, str(e)
            sleep_s = backoff * (2 ** (attempt - 1)) + random.uniform(0, backoff)
            time.sleep(sleep_s)


def discover_ids(query: str, from_date: Optional[str], until_date: Optional[str], max_records: Optional[int], page_size: int, delay_between_pages: float) -> Generator[ArxivEntry, None, None]:
    built_query = build_search_query(query, from_date, until_date)
    start = 0
    total_seen = 0
    total_known = None
    while True:
        entries, total = fetch_arxiv_batch(built_query, start=start, max_results=page_size)
        if total_known is None and total >= 0:
            total_known = total
        if not entries:
            break
        for e in entries:
            yield e
            total_seen += 1
            if max_records is not None and total_seen >= max_records:
                return
        start += len(entries)
        time.sleep(delay_between_pages)


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download arXiv PDFs responsibly")
    parser.add_argument("--query", type=str, default="", help="arXiv API search_query (e.g., 'cat:cs.CL')")
    parser.add_argument("--from-date", type=str, default=None, help="Start date YYYY-MM-DD (submittedDate)")
    parser.add_argument("--until-date", type=str, default=None, help="End date YYYY-MM-DD (submittedDate)")
    parser.add_argument("--out-dir", type=str, default="./arxiv_pdfs", help="Output directory for PDFs")
    parser.add_argument("--manifest", type=str, default=None, help="Path to JSONL manifest (defaults to out-dir/manifest.jsonl)")
    parser.add_argument("--max-records", type=int, default=200, help="Maximum number of records to download")
    parser.add_argument("--max-total-bytes", type=int, default=None, help="Stop after reaching this many bytes")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent download workers")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout seconds per request")
    parser.add_argument("--retries", type=int, default=3, help="Max download retries per file")
    parser.add_argument("--backoff", type=float, default=1.0, help="Base backoff seconds between retries")
    parser.add_argument("--min-pdf-bytes", type=int, default=50000, help="Treat files smaller than this as failed (likely HTML stubs)")
    parser.add_argument("--page-size", type=int, default=100, help="arXiv API page size (<=2000)")
    parser.add_argument("--page-delay", type=float, default=3.0, help="Delay seconds between API pages")
    parser.add_argument("--download-delay", type=float, default=1.0, help="Polite delay seconds between task submissions")
    parser.add_argument("--resume", action="store_true", help="Skip files already present and recorded in manifest")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.jsonl")

    # Load completed IDs if resume
    completed_ids = set()
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                with contextlib.suppress(Exception):
                    rec = json.loads(line)
                    if rec.get("status") == "ok" and rec.get("arxiv_id"):
                        completed_ids.add(rec["arxiv_id"]) 

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    total_downloaded_bytes = 0
    total_attempted = 0
    total_ok = 0
    total_fail = 0

    # Thread pool for downloads
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    active_futures: set = set()
    futures_to_id: dict = {}

    def submit_download(e: ArxivEntry):
        nonlocal total_attempted
        total_attempted += 1
        fut = executor.submit(
            download_pdf,
            e.arxiv_id,
            args.out_dir,
            session,
            args.timeout,
            args.retries,
            args.backoff,
            args.min_pdf_bytes,
        )
        active_futures.add(fut)
        futures_to_id[fut] = e

    def process_future(fut: concurrent.futures.Future) -> bool:
        nonlocal total_ok, total_fail, total_downloaded_bytes
        entry = futures_to_id.pop(fut)
        out_path, ok, size, err = fut.result()
        if ok:
            total_ok += 1
            total_downloaded_bytes += size
            save_manifest_line(
                manifest_path,
                {
                    "arxiv_id": entry.arxiv_id,
                    "title": entry.title,
                    "published": entry.published,
                    "path": os.path.abspath(out_path),
                    "size": size,
                    "status": "ok",
                },
            )
        else:
            total_fail += 1
            save_manifest_line(
                manifest_path,
                {
                    "arxiv_id": entry.arxiv_id,
                    "title": entry.title,
                    "published": entry.published,
                    "path": os.path.abspath(out_path),
                    "size": size,
                    "status": "error",
                    "error": err,
                },
            )
        return ok

    try:
        stop_discovery = False
        for entry in discover_ids(
            query=args.query,
            from_date=args.from_date,
            until_date=args.until_date,
            max_records=args.max_records,
            page_size=args.page_size,
            delay_between_pages=args.page_delay,
        ):
            if args.resume and entry.arxiv_id in completed_ids:
                continue

            if args.max_total_bytes is not None and total_downloaded_bytes >= args.max_total_bytes:
                stop_discovery = True
                break

            # Respect concurrency by waiting for at least one to finish if saturated
            if len(active_futures) >= args.concurrency:
                done_one = next(concurrent.futures.as_completed(active_futures))
                active_futures.remove(done_one)
                process_future(done_one)
                if args.max_total_bytes is not None and total_downloaded_bytes >= args.max_total_bytes:
                    stop_discovery = True
                    break

            submit_download(entry)
            time.sleep(args.download_delay)

        if not stop_discovery:
            # Discovery exhausted; drain remaining futures
            for fut in concurrent.futures.as_completed(active_futures):
                process_future(fut)
                if args.max_total_bytes is not None and total_downloaded_bytes >= args.max_total_bytes:
                    break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        session.close()

    print(
        json.dumps(
            {
                "attempted": total_attempted,
                "ok": total_ok,
                "failed": total_fail,
                "bytes": total_downloaded_bytes,
                "bytes_human": human_bytes(total_downloaded_bytes),
                "out_dir": os.path.abspath(args.out_dir),
                "manifest": os.path.abspath(manifest_path),
            },
            ensure_ascii=False,
        )
    )

    # Return success if at least one file was downloaded or attempted without fatal errors
    return 0


if __name__ == "__main__":
    sys.exit(main())


