#!/usr/bin/env python3
"""
cloud-copy.py

A script to copy data from a Google Cloud Storage bucket/folder
to an AWS S3 bucket/folder using Dask for parallelism on a Coiled cluster.

Features:
  - Recursively lists all files from the GCS source.
  - Creates (or uses) a task file (JSON) to track each file copy's status.
  - Optionally resumes if some files were already copied or failed previously.
  - Gives a rough time estimate and a confirmation prompt before starting.
  - Copies files in parallel, chunk by chunk, to handle large files.
  - Marks tasks as completed or failed, allowing you to retry.

Usage (example):

  python cloud-copy.py \\
      --source-root gs://my-gcs-bucket/data \\
      --dest-root s3://my-s3-bucket/data \\
      --task-file my_tasks.json \\
      --resume

Requires:
  - coiled
  - dask
  - gcsfs
  - s3fs
  - tqdm (optional, but useful for local progress bars)

Authentication:
  - GCS: Set `GOOGLE_APPLICATION_CREDENTIALS` or have gcloud creds in your environment.
  - AWS: Standard environment variables like `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.
"""

import os
import json
import argparse
import coiled
from dask.distributed import Client
import dask.bag as db
import gcsfs
import s3fs
from botocore.exceptions import ClientError


def list_all_files_gcs(source_path: str):
    """
    Recursively list all files in a GCS "directory".
    Returns a list of dicts: [{'src': <full_path>, 'src_size': <bytes>}]
    
    Assumes source_path is either:
        - "gs://bucket/subdir"
        - "bucket/subdir"
    and strips the "gs://" prefix if present.
    """
    gcs_fs = gcsfs.GCSFileSystem()

    # Clean up source_path if it starts with "gs://"
    if source_path.startswith("gs://"):
        source_path = source_path.replace("gs://", "")

    # Ensure we don't have trailing slashes
    source_path = source_path.rstrip("/")
    
    all_files = []
    # fsspec's find returns just files, not directories
    for dirpath, dirs, files in gcs_fs.walk(source_path):
        for fname in files:
            full_path = f"{dirpath}/{fname}"
            info = gcs_fs.info(full_path)
            
            # Only proceed if it's actually a file
            if info.get("type") == "file":
                # This is a file, so you can add it to tasks
                file_size = info.get("size", 0)
                all_files.append({
                    "src": full_path,
                    "src_size": file_size
                })
    return all_files


def s3_connection(aws_profile=None):
    from s3fs import S3FileSystem

    # this will now pick up your “tkt” SSO profile
    fs = S3FileSystem(profile=aws_profile)
    return fs



def generate_task_file(source_files, src_root, dst_root, task_filename="copy_tasks.json", aws_profile = "trazo", check_destination=True):
    """
    Create a JSON 'task file' with entries for each file:
      {
        "src": <GCS path>,
        "dst": <S3 path>,
        "src_size": <bytes>,
        "dst_size": <bytes>,
        "status": "pending"
      }
    
    The dst path is computed by replacing 'src_root' in the file's path
    with 'dst_root' (keeping the relative path).
    
    Returns the list of tasks.
    """
    tasks = []
    
    # Ensure we don't have trailing slashes
    src_root = src_root.rstrip("/")
    dst_root = dst_root.rstrip("/")

    # Create an s3fs filesystem if we're checking existing files
    if check_destination:
        s3_fs = s3_connection(aws_profile=aws_profile)

    counter = 0
    for item in source_files:
        counter += 1
        src_path = item["src"]
        src_size = item["src_size"]

        # Compute relative path
        rel_path = src_path[len(src_root):].lstrip("/")

        # Compute S3 destination path
        dst_path = f'{dst_root}/{rel_path}'

        # By default, assume status is pending
        status = "pending"
        dst_size = None
        error = None

        if check_destination:
            # Check if the file already exists in S3
            try:
                info = s3_fs.info(dst_path)  # will raise FileNotFoundError if not present
                dst_size = info.get("size", 0)
                
                # If sizes match, mark it as completed
                if dst_size == src_size:
                    status = "completed"
                else:
                    status = "mismatched"
            except FileNotFoundError:
                # Object not found, so it's definitely pending
                pass
            except ClientError as e:
                # Some other error occurred (e.g. permissions)
                status = "failed"
                error = str(e)

        # Add the task entry
        tasks.append({
            "src": src_path,
            "dst": dst_path,
            "src_size": src_size,
            "dst_size": dst_size,
            "status": status,
            "error": error
        })

        # Print progess on a single line
        print(f'Check existing: {src_path} ({status}) {counter}', end="\r")
    print("\n")

    
    with open(task_filename, "w") as f:
        json.dump(tasks, f, indent=2)
    
    return tasks

def load_task_file(task_filename):
    """
    Load a JSON task file, returning only tasks
    that are pending or failed.
    """
    with open(task_filename, "r") as f:
        tasks = json.load(f)
    return [t for t in tasks if t["status"] in ("pending", "failed")]

def copy_file(task, aws_profile = "trazo"):
    """
    Copy one file from GCS to S3 using fsspec-based filesystems.
    Return the task dict with updated status.
    """
    src_path = task["src"]
    dst_path = task["dst"]
    file_size = task["src_size"]  # from the task file

    gcs_fs = gcsfs.GCSFileSystem()
    s3_fs = s3_connection(aws_profile=aws_profile)

    try:
        # 1. Check if file already exists in S3
        if s3_fs.exists(dst_path):
            info = s3_fs.info(dst_path)
            # 2. Compare sizes (or timestamps, checksums, etc.)
            if info.get("size", 0) == file_size:
                # Mark as 'completed' (skipped copy)
                task["status"] = "completed"
                task["skipped"] = True
                return task
            # else: proceed to overwrite or re-copy

        # 3. If not exist or size mismatch, do the actual copy
        with gcs_fs.open(src_path, "rb") as src_f:
            with s3_fs.open(dst_path, "wb") as dst_f:
                chunk_size = 1024 * 1024
                while True:
                    chunk = src_f.read(chunk_size)
                    if not chunk:
                        break
                    dst_f.write(chunk)
        
        # Get the size of the newly written object
        dst_info = s3_fs.info(dst_path)
        task["dst_size"] = dst_info.get("size", 0)

        if file_size != task["dst_size"]:
            raise ValueError(f'Mismatch in file size for {dst_path}. '
                             f'Expected {file_size}, got {task["dst_size"]}')

        task["status"] = "completed"
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)

    return task

def save_progress(task_filename, updated_tasks, all_tasks):
    """
    Merge statuses from 'updated_tasks' into 'all_tasks',
    then write back to the JSON file.
    """
    # Convert all_tasks to a dict by 'src' for quick lookup
    task_map = {t["src"]: t for t in all_tasks}
    
    for ut in updated_tasks:
        src_key = ut["src"]
        task_map[src_key]["status"] = ut["status"]

        if "error" in ut:
            task_map[src_key]["error"] = ut["error"]

        # If a dst_size is present, store it in the master list
        if "dst_size" in ut:
            task_map[src_key]["dst_size"] = ut["dst_size"]
    
    # Convert back to a list
    updated_list = list(task_map.values())
    
    # Save to disk
    with open(task_filename, "w") as f:
        json.dump(updated_list, f, indent=2)

def estimate_time(total_bytes, concurrency, rate_per_worker=50e6):
    """
    Roughly estimate the time to transfer total_bytes,
    given concurrency and an assumed rate (in bytes/sec) per worker.
    Default: 50 MB/s per worker.
    Returns the time (seconds).
    """
    if concurrency <= 0:
        concurrency = 1
    return total_bytes / (concurrency * rate_per_worker)

def main():
    parser = argparse.ArgumentParser(
        description="Copy data from GCS to S3 using Dask + Coiled."
    )
    parser.add_argument("--source-root", required=True,
                        help="GCS bucket/folder (e.g., gs://my-bucket/data).")
    parser.add_argument("--dest-root", required=True,
                        help="S3 bucket/folder (e.g., s3://my-bucket/data).")
    parser.add_argument("--task-file",
                        help="File to store tasks (JSON). Default: cloudcopy_tasks.json")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing task file, skipping completed tasks.")
    parser.add_argument("--coiled-region", default="us-east-1",
                        help="Coiled cluster region. Default: us-east-1")
    parser.add_argument("--aws-profile", default="trazo",
                        help="AWS profile to use for S3 access. Default: None (uses default credentials).")
    
    args = parser.parse_args()
    
    # Clean up user-provided paths
    src_root = args.source_root.replace("gs://", "")
    dst_root = args.dest_root.replace("s3://", "")

    # Check if we have access to s3
    s3_fs = s3_connection(aws_profile=args.aws_profile)
    s3_fs.ls(dst_root)
    
    
    # 1. Create or load tasks
    if not args.resume:
        print(f'Listing all files in GCS path: gs://{src_root} ...')
        files = list_all_files_gcs(src_root)
        total_size = sum(f["src_size"] for f in files)
        print(f'  Found {len(files)} files, total size = {total_size/1e9:.2f} GB.\n')
        
        tasks = generate_task_file(
            source_files=files,
            src_root=src_root,
            dst_root=dst_root,
            task_filename=args.task_file,
            aws_profile = "trazo"
        )
        tasks = load_task_file(args.task_file)
        total_size = sum(t["src_size"] for t in tasks)
        print(f'  Pending/failed tasks: {len(tasks)}, total size = {total_size/1e9:.2f} GB.\n')

    else:
        print(f'Resuming from existing task file: {args.task_file}')
        tasks = load_task_file(args.task_file)
        total_size = sum(t["src_size"] for t in tasks)
        print(f'  Pending/failed tasks: {len(tasks)}, total size = {total_size/1e9:.2f} GB.\n')
    
    # 2. Provide a rough time estimate
    concurrency_guess = 10
    estimated_seconds = estimate_time(total_size, concurrency_guess)
    estimated_minutes = estimated_seconds / 60
    print(f'Estimated time (rough): {estimated_minutes:.1f} minutes (assuming ~{concurrency_guess} workers).')
    
    # 3. Confirmation prompt
    choice = input("Continue with the copy? [y/N]: ")
    if choice.lower() not in ("y", "yes"):
        print("Aborting.")
        return
    
    
    # 5. Create a Dask Bag of tasks and run
    print("\nStarting the copy operations in parallel...")
    bag = db.from_sequence(tasks, npartitions=len(tasks))
    results = bag.map(copy_file).compute()
    
    # 6. Save final statuses back to the task file
    #    We need the entire list of tasks (including completed ones).
    #    The simplest approach is to read them all back and re-merge.
    with open(args.task_file, "r") as f:
        all_tasks = json.load(f)
    
    save_progress(args.task_file, results, all_tasks)
    
    # Summarize
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f'\nCopy completed! {completed} succeeded, {failed} failed.')
    if failed > 0:
        print("You can re-run with --resume to retry the failed tasks.")
    

if __name__ == "__main__":
    main()
