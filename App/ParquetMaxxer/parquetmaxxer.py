import pandas as pd
import numpy as np
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa

# Set output dir globally (will be created when needed)
OUTPUT_DIR = "./converted"

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
def convert_jsonl_to_parquet(file_path, chunk_size=10000):
    """
    Convert a JSONL file to a Parquet file in chunks, writing each chunk directly.
    Outputs to ./converted directory.
    Returns:
        (out_path, preview): Path to output parquet, and a preview string of first rows.
    """
    # Create output directory only when actually converting
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(OUTPUT_DIR, base_name + ".parquet")
    try:
        df_iter = pd.read_json(file_path, lines=True, chunksize=chunk_size)
    except ValueError as e:
        raise RuntimeError(f"Failed to read JSONL: {e}")

    first_chunk = True
    preview_df = None
    writer = None
    try:
        for chunk in df_iter:
            if preview_df is None:
                preview_df = chunk.head(5)
            table = pa.Table.from_pandas(chunk)
            if first_chunk:
                # Initialize ParquetWriter for the first chunk
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writer.write_table(table)
                first_chunk = False
            else:
                # Append subsequent chunks
                writer.write_table(table)
    finally:
        # Ensure the writer is closed
        if writer is not None:
            writer.close()

    if preview_df is None:
        # File empty case: write empty parquet
        empty_df = pd.DataFrame()
        pq.write_table(pa.Table.from_pandas(empty_df), out_path, compression="snappy")
        preview = "(empty file)"
    else:
        preview = preview_df.to_string()
    return out_path, preview

def convert_parquet_to_jsonl(file_path):
    """
    Convert a Parquet file to a JSONL file.
    Outputs to ./converted directory.
    Returns:
        (out_path, preview): Path to output JSONL, and a preview string of first rows.
    """
    # Create output directory only when actually converting
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(OUTPUT_DIR, base_name + ".jsonl")
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
        queue.put((file_path, out_path, preview, None))
    except Exception as e:
        queue.put((file_path, None, None, str(e)))

def parquet_to_jsonl_worker(file_path, queue):
    """
    Worker for multiprocessing. Converts Parquet to JSONL and puts result in queue.
    Queue receives (input_path, preview, error).
    """
    try:
        out_path, preview = convert_parquet_to_jsonl(file_path)
        queue.put((file_path, out_path, preview, None))
    except Exception as e:
        queue.put((file_path, None, None, str(e)))