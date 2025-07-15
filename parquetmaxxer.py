import pandas as pd
import numpy as np
import json
import os

# ---------- Utility ----------
def sanitize_for_json(obj):
    """
    Recursively sanitize an object for JSON serialization:
    - Convert numpy types to native Python.
    - Replace NaN with None.
    - Handle lists, dicts, and arrays deeply.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj


# ---------- Core conversions ----------
def convert_jsonl_to_parquet(file_path, chunk_size=1000):
    """
    Convert a JSONL file to a Parquet file.

    Parameters:
        file_path: Path to the input .jsonl file.
        chunk_size: Rows per chunk for memory efficiency.

    Returns:
        (out_path, preview): Path to output parquet, and a preview string of first rows.
    """
    base, _ = os.path.splitext(file_path)
    out_path = base + ".parquet"
    all_chunks = []

    # Read in chunks to handle large files
    try:
        df_iter = pd.read_json(file_path, lines=True, chunksize=chunk_size)
        for chunk in df_iter:
            all_chunks.append(chunk)
    except ValueError as e:
        # Invalid JSONL or empty file
        raise RuntimeError(f"Failed to read JSONL: {e}")

    if not all_chunks:
        # Empty file
        pd.DataFrame().to_parquet(out_path, index=False, compression="snappy")
        return out_path, "(empty file)"

    df = pd.concat(all_chunks, ignore_index=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    preview = df.head(5).to_string()
    return out_path, preview


def convert_parquet_to_jsonl(file_path):
    """
    Convert a Parquet file to a JSONL file.

    Parameters:
        file_path: Path to the input .parquet file.

    Returns:
        (out_path, preview): Path to output JSONL, and a preview string of first rows.
    """
    out_path = os.path.splitext(file_path)[0] + ".jsonl"
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read Parquet: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            json_line = json.dumps(sanitize_for_json(record), ensure_ascii=False)
            f.write(json_line + "\n")

    preview = df.head(5).to_string()
    return out_path, preview


# ---------- Worker entry points ----------
def jsonl_to_parquet_worker(file_path, queue):
    """
    Worker for multiprocessing. Converts JSONL to Parquet and puts result in queue.
    Queue receives (input_path, preview, error).
    """
    try:
        out_path, preview = convert_jsonl_to_parquet(file_path)
        queue.put((file_path, preview, None))
    except Exception as e:
        queue.put((file_path, None, str(e)))


def parquet_to_jsonl_worker(file_path, queue):
    """
    Worker for multiprocessing. Converts Parquet to JSONL and puts result in queue.
    Queue receives (input_path, preview, error).
    """
    try:
        out_path, preview = convert_parquet_to_jsonl(file_path)
        queue.put((file_path, preview, None))
    except Exception as e:
        queue.put((file_path, None, str(e)))
