import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synchronize files from a source directory to a target directory."
    )
    parser.add_argument("source", type=str, help="Path to the source directory")
    parser.add_argument("target", type=str, help="Path to the target directory")
    parser.add_argument(
        "--mode",
        choices=["skip", "overwrite"],
        default="skip",
        help="How to handle existing files in the target directory (default: skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files.",
    )
    return parser.parse_args()


def sync_directories(source: Path, target: Path, mode: str = "skip", dry_run: bool = False):
    if not source.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source}")

    # First pass: collect all files to get total count
    all_files = []
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        rel_path = root_path.relative_to(source)
        target_root = target / rel_path
        
        for f in files:
            src_file = root_path / f
            dst_file = target_root / f
            all_files.append((src_file, dst_file, root_path, target_root))

    copied = 0
    skipped = 0
    overwritten = 0

    # Create progress bar
    with tqdm(total=len(all_files), desc="Syncing files", unit="files") as pbar:
        for src_file, dst_file, root_path, target_root in all_files:
            # Ensure directories exist in target
            if not dry_run:
                target_root.mkdir(parents=True, exist_ok=True)

            if dst_file.exists():
                if mode == "skip":
                    skipped += 1
                    pbar.set_postfix({"copied": copied, "skipped": skipped, "overwritten": overwritten})
                    pbar.update(1)
                    continue
                elif mode == "overwrite":
                    overwritten += 1
            else:
                copied += 1

            if not dry_run:
                shutil.copy2(src_file, dst_file)
            
            pbar.set_postfix({"copied": copied, "skipped": skipped, "overwritten": overwritten})
            pbar.update(1)

    print(f"\nSummary:")
    print(f"  Copied new files   : {copied}")
    print(f"  Overwritten files  : {overwritten}")
    print(f"  Skipped files      : {skipped}")


def main():
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()

    print(f"Synchronizing files:\n  Source : {source}\n  Target : {target}\n  Mode   : {args.mode}\n  DryRun : {args.dry_run}")
    sync_directories(source, target, mode=args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

# python 1_copy_md_label.py /home/suny0a/arxiv_dataset_ibex/reasoning \
#                           /home/suny0a/blob/yasheng/arxiv_dataset/reasoning
