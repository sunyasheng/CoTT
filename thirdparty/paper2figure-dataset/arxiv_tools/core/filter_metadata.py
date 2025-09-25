#!/usr/bin/env python3
"""
Filter arXiv metadata snapshot and produce:
- output/paper_ids.txt                (one arXiv id per line)
- output/filtered_papers_metadata.xlsx (tabular metadata)

Inputs:
- A newline-delimited JSON snapshot file like arxiv-metadata-oai-snapshot.json

Filtering options (optional):
- --year YEAR                     Only include first-submitted year == YEAR
- --start_year Y1 --end_year Y2  Include if Y1 <= year <= Y2 (inclusive)
- --categories cat1,cat2         Comma-separated list; include if any category matches
- --limit N                       Stop after N records that match
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def parse_year(record: Dict) -> Optional[int]:
    versions = record.get("versions") or []
    if versions:
        created = versions[0].get("created")
        # created example: "2020-01-01 00:00:00"
        if isinstance(created, str) and len(created) >= 4 and created[:4].isdigit():
            return int(created[:4])
    # fallback to update_date like "2020-01-01"
    update_date = record.get("update_date")
    if isinstance(update_date, str) and len(update_date) >= 4 and update_date[:4].isdigit():
        return int(update_date[:4])
    return None


def record_matches(record: Dict, year: Optional[int], categories: Optional[List[str]], start_year: Optional[int] = None, end_year: Optional[int] = None) -> bool:
    ry = parse_year(record)
    if year is not None:
        if ry != year:
            return False
    else:
        if start_year is not None or end_year is not None:
            if ry is None:
                return False
            if start_year is not None and ry < start_year:
                return False
            if end_year is not None and ry > end_year:
                return False

    if categories:
        rec_cats = (record.get("categories") or "").split()
        if not any(cat in rec_cats for cat in categories):
            return False

    return True


def build_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def filter_snapshot(
    snapshot_path: Path,
    output_dir: Path,
    year: Optional[int],
    categories: Optional[List[str]],
    limit: Optional[int],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_ids: List[str] = []
    rows: List[Dict] = []

    with snapshot_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_id = rec.get("id") or rec.get("paper_id")
            if not paper_id:
                continue

            if not record_matches(rec, year, categories, start_year, end_year):
                continue

            pdf_url = build_pdf_url(paper_id)
            title = rec.get("title")
            authors = "; ".join(a.get("name") for a in (rec.get("authors") or []) if isinstance(a, dict) and a.get("name")) if isinstance(rec.get("authors"), list) else rec.get("authors")
            primary_category = (rec.get("categories") or "").split()[0] if rec.get("categories") else None
            year_val = parse_year(rec)

            paper_ids.append(paper_id)
            rows.append({
                "id": paper_id,
                "title": title,
                "authors": authors,
                "categories": rec.get("categories"),
                "primary_category": primary_category,
                "year": year_val,
                "pdf_url": pdf_url,
            })

            if limit is not None and len(paper_ids) >= limit:
                break

    # Write paper ids
    ids_path = output_dir / "paper_ids.txt"
    with ids_path.open("w") as f:
        for pid in paper_ids:
            f.write(pid + "\n")

    # Write metadata table (use openpyxl to avoid xlsxwriter URL auto-link warnings)
    df = pd.DataFrame(rows)
    xlsx_path = output_dir / "filtered_papers_metadata.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    print("✅ Filter complete")
    print(f"   📄 IDs: {ids_path}")
    print(f"   📊 Metadata: {xlsx_path}")
    print(f"   🧮 Count: {len(paper_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter arXiv metadata snapshot and export IDs + metadata table")
    parser.add_argument("--snapshot", required=True, help="Path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--output", default="./output", help="Output directory (will be created)")
    parser.add_argument("--year", type=int, help="Only include records whose first submission year == YEAR")
    parser.add_argument("--start_year", type=int, help="Include records with year >= START_YEAR")
    parser.add_argument("--end_year", type=int, help="Include records with year <= END_YEAR")
    parser.add_argument("--categories", help="Comma-separated list of categories to include (match any)")
    parser.add_argument("--limit", type=int, help="Stop after N matching records")

    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(',')] if args.categories else None

    filter_snapshot(
        snapshot_path=Path(args.snapshot),
        output_dir=Path(args.output),
        year=args.year,
        categories=categories,
        limit=args.limit,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()


