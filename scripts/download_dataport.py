"""
#!/usr/bin/env python3
"""

"""
Download a CRAWDAD dataport archive (or any URL), extract relevant files, and create a small sample CSV for testing.

Usage:
    python scripts/download_dataport.py --url <DATA_URL> --outdir data --create-sample --sample-rows 100

This script does NOT commit large datasets to the repo. It downloads to the specified outdir and can optionally create a small sampled CSV that is safe to commit.

Note: check license of the dataport before redistributing.
"""

import argparse
import os
import sys
import shutil
import tempfile
import csv
from urllib.parse import urlparse

try:
    import requests
except Exception:
    requests = None

import tarfile
import zipfile

def download_file(url, dest_path, chunk_size=8192):
    if requests is None:
        raise RuntimeError("requests is required to download files. Install with: pip install requests")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = r.headers.get('content-length')
        total = int(total) if total is not None else None
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest_path

def safe_extract_archive(archive_path, extract_to):
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(path=extract_to)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(path=extract_to)
    else:
        raise RuntimeError('Unknown archive format for: %s' % archive_path)

def sample_csv_from_dir(src_dir, out_csv_path, sample_rows=200, max_files=50):
    """Search for CSV-like files in src_dir and concatenate a sample into out_csv_path.
    This is a best-effort helper for creating a small sample dataset for the notebook/tests.
    """
    candidates = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if fname.lower().endswith(('.csv', '.txt')):
                candidates.append(os.path.join(root, fname))
                if len(candidates) >= max_files:
                    break
    if not candidates:
        raise RuntimeError('No CSV/text files found in extracted dataport to sample from.')

    rows_written = 0
    header = None
    with open(out_csv_path, 'w', newline='') as out_f:
        writer = None
        for fpath in candidates:
            try:
                with open(fpath, 'r', errors='ignore') as in_f:
                    reader = csv.reader(in_f)
                    local_header = None
                    for i, row in enumerate(reader):
                        if i == 0 and header is None:
                            local_header = row
                            header = row
                            writer = csv.writer(out_f)
                            writer.writerow(header)
                            continue
                        if header is None:
                            # skip until we find a header
                            continue
                        # If row length mismatches header, try to skip
                        if len(row) != len(header):
                            continue
                        writer.writerow(row)
                        rows_written += 1
                        if rows_written >= sample_rows:
                            break
            except Exception:
                # ignore problematic files
                continue
            if rows_written >= sample_rows:
                break
    if rows_written == 0:
        raise RuntimeError('Failed to write any sample rows from dataport files.')
    return out_csv_path

def main():
    parser = argparse.ArgumentParser(description='Download dataport archive and create small sample CSV')
    parser.add_argument('--url', type=str, required=True, help='URL of the dataport archive to download (e.g., a CRAWDAD dataport file)')
    parser.add_argument('--outdir', type=str, default='data', help='Directory to download/extract into')
    parser.add_argument('--create-sample', action='store_true', help='Create a small sampled CSV from the extracted files')
    parser.add_argument('--sample-rows', type=int, default=200, help='Number of rows to include in the sample csv')
    parser.add_argument('--keep-archive', action='store_true', help='Do not delete downloaded archive after extraction')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    parsed = urlparse(args.url)
    fname = os.path.basename(parsed.path) or 'dataport_download'
    tmp_dir = tempfile.mkdtemp(prefix='dataport_')
    archive_path = os.path.join(tmp_dir, fname)

    try:
        print('Downloading', args.url)
        download_file(args.url, archive_path)
        print('Downloaded to', archive_path)

        print('Extracting archive...')
        safe_extract_archive(archive_path, tmp_dir)
        print('Extraction complete. Searching for CSV/text files to sample...')

        if args.create_sample:
            out_csv = os.path.join(args.outdir, 'sample_dataport_sample.csv')
            sample_csv_from_dir(tmp_dir, out_csv, sample_rows=args.sample_rows)
            print('Sample CSV written to', out_csv)

        if not args.keep_archive:
            try:
                os.remove(archive_path)
            except Exception:
                pass

    finally:
        # keep the extracted dir for user inspection in outdir/extracted_dataport
        extracted_dir = os.path.join(args.outdir, 'extracted_dataport')
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir, ignore_errors=True)
        shutil.move(tmp_dir, extracted_dir)
        print('Moved extracted files to', extracted_dir)


if __name__ == '__main__':
    main()
