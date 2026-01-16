import os
import sys
import json
import hashlib
import logging
import contextlib
import tempfile
import shutil
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageOps

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[]
)

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def default_update_status(_: str):
    pass

def default_update_progress(_: int, __: int):
    pass


@dataclass
class ImageMeta:
    path: str
    size_bytes: int
    width: int
    height: int
    fmt: str


class ImageDeduplication:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

    def __init__(
        self,
        dhash_hamming_threshold: int = 8,
        prefix_bits: int = 16,
        embedding_threshold: float = 0.97,
        embed_model_name: str = "openai/clip-vit-base-patch32",
        text_sim_threshold: float = 0.85,
        text_embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.dhash_hamming_threshold = dhash_hamming_threshold
        self.prefix_bits = prefix_bits
        self.embedding_threshold = embedding_threshold
        self.embed_model_name = embed_model_name
        self.text_sim_threshold = text_sim_threshold
        self.text_embed_model = text_embed_model

    def iter_image_paths(self, inputs: List[str]) -> List[str]:
        paths: List[str] = []
        seen = set()

        for raw in inputs:
            p = Path(raw)
            if p.is_dir():
                for root, _, files in os.walk(p):
                    for fn in files:
                        ext = Path(fn).suffix.lower()
                        if ext in self.IMAGE_EXTS:
                            full = str(Path(root) / fn)
                            if full not in seen:
                                seen.add(full)
                                paths.append(full)
            elif p.is_file():
                if p.suffix.lower() in self.IMAGE_EXTS:
                    full = str(p)
                    if full not in seen:
                        seen.add(full)
                        paths.append(full)
        return paths

    def load_parquet_images(
        self,
        parquet_paths: List[str],
        output_dir: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        image_column: str = "image",
        text_column: str = "text",
        prompt_column: str = "prompt",
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load images from parquet files and extract them to a directory.
        
        Parquet format expected (HuggingFace image dataset style):
            - image: {"bytes": <image bytes>, "path": <relative path>}
            - text: caption/description string
            - prompt: prompt string (optional)
        
        Args:
            parquet_paths: List of parquet file paths to load
            output_dir: Directory to extract images to
            image_column: Column name containing image data (default: "image")
            text_column: Column name containing text/caption (default: "text")
            prompt_column: Column name containing prompt (default: "prompt")
        
        Returns:
            Tuple of:
                - List of extracted image paths
                - Dict mapping image path -> metadata dict (text, prompt, source_parquet, row_idx)
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            update_status("❌ pyarrow is required. Install with: pip install pyarrow")
            return [], {}
        
        self._ensure_dir(output_dir)
        
        image_paths: List[str] = []
        metadata_map: Dict[str, Dict[str, Any]] = {}
        total_rows = 0
        processed_rows = 0
        
        # First pass: count total rows
        update_status(f"Scanning {len(parquet_paths)} parquet file(s)...")
        for pq_path in parquet_paths:
            try:
                pf = pq.ParquetFile(pq_path)
                total_rows += pf.metadata.num_rows
            except Exception as e:
                update_status(f"⚠️ Error reading {os.path.basename(pq_path)}: {e}")
        
        if total_rows == 0:
            update_status("❌ No rows found in parquet files.")
            return [], {}
        
        update_status(f"Extracting {total_rows} images from parquet files...")
        update_progress(0, total_rows)
        
        # Second pass: extract images
        global_idx = 0
        for pq_path in parquet_paths:
            try:
                pq_name = os.path.basename(pq_path)
                table = pq.read_table(pq_path)
                
                # Get column data
                if image_column not in table.column_names:
                    update_status(f"⚠️ No '{image_column}' column in {pq_name}, skipping...")
                    continue
                
                num_rows = table.num_rows
                
                for row_idx in range(num_rows):
                    try:
                        # Extract image data
                        img_data = table[image_column][row_idx].as_py()
                        
                        if img_data is None:
                            processed_rows += 1
                            global_idx += 1
                            continue
                        
                        # Handle different image formats in parquet
                        img_bytes = None
                        img_rel_path = None
                        
                        if isinstance(img_data, dict):
                            # Standard HuggingFace format: {"bytes": ..., "path": ...}
                            img_bytes = img_data.get("bytes")
                            img_rel_path = img_data.get("path", "")
                        elif isinstance(img_data, bytes):
                            # Direct bytes storage
                            img_bytes = img_data
                            img_rel_path = ""
                        elif hasattr(img_data, 'tobytes'):
                            # PIL Image or similar
                            img_bytes = img_data.tobytes()
                            img_rel_path = ""
                        
                        if img_bytes is None:
                            processed_rows += 1
                            global_idx += 1
                            continue
                        
                        # Determine output filename
                        if img_rel_path:
                            # Use path from parquet, but flatten subdirs with underscores
                            safe_name = img_rel_path.replace("/", "_").replace("\\", "_")
                        else:
                            # Generate filename based on index
                            safe_name = f"parquet_{global_idx:08d}.png"
                        
                        # Ensure unique filename
                        out_path = os.path.join(output_dir, safe_name)
                        if os.path.exists(out_path):
                            stem, ext = os.path.splitext(safe_name)
                            k = 1
                            while os.path.exists(out_path):
                                out_path = os.path.join(output_dir, f"{stem}_{k}{ext}")
                                k += 1
                        
                        # Detect image format and save
                        try:
                            img = Image.open(io.BytesIO(img_bytes))
                            img = ImageOps.exif_transpose(img)
                            
                            # Ensure we have a proper extension
                            if not out_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')):
                                fmt = img.format or "PNG"
                                ext = f".{fmt.lower()}"
                                if ext == ".jpeg":
                                    ext = ".jpg"
                                out_path = out_path + ext
                            
                            # Save image
                            img.save(out_path)
                            img.close()
                        except Exception:
                            # Fallback: write raw bytes
                            with open(out_path, 'wb') as f:
                                f.write(img_bytes)
                        
                        # Collect metadata
                        text_val = ""
                        if text_column in table.column_names:
                            try:
                                text_val = table[text_column][row_idx].as_py() or ""
                            except Exception:
                                pass
                        
                        prompt_val = ""
                        if prompt_column in table.column_names:
                            try:
                                prompt_val = table[prompt_column][row_idx].as_py() or ""
                            except Exception:
                                pass
                        
                        image_paths.append(out_path)
                        metadata_map[out_path] = {
                            "text": text_val,
                            "prompt": prompt_val,
                            "source_parquet": pq_name,
                            "row_idx": row_idx,
                            "original_path": img_rel_path,
                        }
                        
                    except Exception as e:
                        logging.error(f"Error extracting row {row_idx} from {pq_name}: {e}")
                    
                    processed_rows += 1
                    global_idx += 1
                    
                    if processed_rows % 100 == 0:
                        update_progress(processed_rows, total_rows)
                
            except Exception as e:
                logging.error(f"Error processing parquet {pq_path}: {e}", exc_info=True)
                update_status(f"⚠️ Error processing {os.path.basename(pq_path)}: {e}")
        
        update_progress(total_rows, total_rows)
        update_status(f"✅ Extracted {len(image_paths)} images from {len(parquet_paths)} parquet file(s)")
        
        return image_paths, metadata_map

    def write_deduplicated_parquet(
        self,
        kept_paths: List[str],
        metadata_map: Dict[str, Dict[str, Any]],
        output_path: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        compression: str = "zstd",
        randomize: bool = False,
    ) -> int:
        """
        Write deduplicated images back to a parquet file in HuggingFace format.
        
        Args:
            kept_paths: List of image paths to keep (after deduplication)
            metadata_map: Dict mapping image path -> metadata dict
            output_path: Output parquet file path
            compression: Parquet compression (zstd, snappy, gzip, none)
            randomize: If True, shuffle the output order
        
        Returns:
            Number of rows written
        """
        import random
        
        try:
            from datasets import Dataset, Features, Value, Image as HFImage
        except ImportError:
            update_status("❌ datasets library required. Install with: pip install datasets")
            return 0
        
        # Randomize order if requested
        if randomize:
            kept_paths = kept_paths.copy()
            random.shuffle(kept_paths)
            update_status("Shuffled output order")
        
        update_status(f"Writing {len(kept_paths)} images to parquet...")
        update_progress(0, len(kept_paths))
        
        image_items = []
        text_list = []
        prompt_list = []
        
        for i, img_path in enumerate(kept_paths):
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                
                # Get original path from metadata if available
                meta = metadata_map.get(img_path, {})
                rel_path = meta.get("original_path") or os.path.basename(img_path)
                
                image_items.append({"bytes": img_bytes, "path": rel_path})
                text_list.append(meta.get("text", ""))
                prompt_list.append(meta.get("prompt", ""))
                
            except Exception as e:
                logging.error(f"Error reading {img_path}: {e}")
            
            if i % 100 == 0:
                update_progress(i, len(kept_paths))
        
        features = Features({
            "image": HFImage(),
            "text": Value("string"),
            "prompt": Value("string"),
        })
        
        data = {"image": image_items, "text": text_list, "prompt": prompt_list}
        ds = Dataset.from_dict(data)
        ds = ds.cast(features)
        
        compression_arg = None if compression == "none" else compression
        ds.to_parquet(output_path, compression=compression_arg)
        
        update_progress(len(kept_paths), len(kept_paths))
        update_status(f"✅ Wrote {len(kept_paths)} images to {os.path.basename(output_path)}")
        
        return len(kept_paths)

    def load_hf_dataset_images(
        self,
        dataset_name: str,
        output_dir: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        image_column: str = "image",
        text_column: str = "text",
        split: Optional[str] = None,
        token: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load images from a HuggingFace dataset and extract them to a directory.
        
        Supports various dataset formats:
            - Image datasets with 'image' column (PIL images)
            - Text-image pairs with 'image' and 'text'/'caption' columns
            - Multimodal datasets
        
        Args:
            dataset_name: HuggingFace dataset name (e.g., 'org/dataset_name')
            output_dir: Directory to extract images to
            image_column: Column name containing images (default: 'image')
            text_column: Column name containing text/captions (default: 'text')
            split: Dataset split to use (default: first available split)
            token: HuggingFace token for private datasets
            max_samples: Maximum number of samples to load (None = all)
        
        Returns:
            Tuple of:
                - List of extracted image paths
                - Dict mapping image path -> metadata dict (text, source, row_idx)
        """
        try:
            from datasets import load_dataset, DatasetDict, IterableDataset
        except ImportError:
            update_status("❌ datasets library required. Install with: pip install datasets")
            return [], {}
        
        self._ensure_dir(output_dir)
        
        image_paths: List[str] = []
        metadata_map: Dict[str, Dict[str, Any]] = {}
        
        # Load dataset
        update_status(f"🤗 Loading HuggingFace dataset: {dataset_name}...")
        
        try:
            token_param = token if token else None
            
            if split:
                dataset = load_dataset(dataset_name, split=split, token=token_param)
                update_status(f"Using split: {split}")
            else:
                dataset_dict = load_dataset(dataset_name, token=token_param)
                if isinstance(dataset_dict, DatasetDict):
                    # Use first available split
                    split_name = list(dataset_dict.keys())[0]
                    dataset = dataset_dict[split_name]
                    update_status(f"Using split: {split_name}")
                else:
                    dataset = dataset_dict
        except Exception as e:
            update_status(f"❌ Failed to load dataset: {e}")
            return [], {}
        
        # Get dataset size
        try:
            if isinstance(dataset, IterableDataset):
                total_samples = max_samples if max_samples else 100000  # Estimate for iterable
                update_status(f"Dataset is iterable (streaming mode)")
            else:
                total_samples = len(dataset)
                if max_samples:
                    total_samples = min(total_samples, max_samples)
                update_status(f"Dataset has {len(dataset)} samples" + (f" (loading {total_samples})" if max_samples else ""))
        except (TypeError, AttributeError):
            total_samples = max_samples if max_samples else 10000
        
        # Check available columns
        try:
            if hasattr(dataset, 'column_names'):
                columns = dataset.column_names
                update_status(f"Available columns: {', '.join(columns)}")
                
                # Auto-detect image column
                if image_column not in columns:
                    for col in ['image', 'img', 'picture', 'photo']:
                        if col in columns:
                            image_column = col
                            update_status(f"Using image column: {image_column}")
                            break
                
                # Auto-detect text column
                if text_column not in columns:
                    for col in ['text', 'caption', 'description', 'prompt', 'label']:
                        if col in columns:
                            text_column = col
                            update_status(f"Using text column: {text_column}")
                            break
        except Exception:
            pass
        
        update_status(f"Extracting images to {output_dir}...")
        update_progress(0, total_samples)
        
        # Extract images
        extracted = 0
        for idx, sample in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            
            try:
                # Get image
                img_data = sample.get(image_column)
                if img_data is None:
                    continue
                
                # Handle different image formats
                img = None
                if hasattr(img_data, 'save'):
                    # PIL Image
                    img = img_data
                elif isinstance(img_data, dict) and 'bytes' in img_data:
                    # HuggingFace Image format with bytes
                    img = Image.open(io.BytesIO(img_data['bytes']))
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                elif isinstance(img_data, str) and os.path.exists(img_data):
                    # File path
                    img = Image.open(img_data)
                
                if img is None:
                    continue
                
                # Determine output filename
                out_name = f"hf_{idx:08d}.png"
                out_path = os.path.join(output_dir, out_name)
                
                # Handle collisions
                if os.path.exists(out_path):
                    k = 1
                    while os.path.exists(out_path):
                        out_path = os.path.join(output_dir, f"hf_{idx:08d}_{k}.png")
                        k += 1
                
                # Save image
                img = ImageOps.exif_transpose(img)
                img.save(out_path)
                if hasattr(img, 'close'):
                    img.close()
                
                # Get text/caption
                text_val = ""
                if text_column:
                    text_val = sample.get(text_column, "") or ""
                    if not isinstance(text_val, str):
                        text_val = str(text_val)
                
                # Store additional metadata columns
                extra_meta = {}
                for key, val in sample.items():
                    if key not in [image_column, text_column] and isinstance(val, (str, int, float, bool)):
                        extra_meta[key] = val
                
                image_paths.append(out_path)
                metadata_map[out_path] = {
                    "text": text_val,
                    "source_dataset": dataset_name,
                    "row_idx": idx,
                    **extra_meta,
                }
                extracted += 1
                
            except Exception as e:
                logging.error(f"Error extracting sample {idx}: {e}")
            
            if idx % 100 == 0:
                update_progress(idx, total_samples)
        
        update_progress(total_samples, total_samples)
        update_status(f"✅ Extracted {extracted} images from HuggingFace dataset")
        
        return image_paths, metadata_map

    @staticmethod
    def _safe_stat_size(path: str) -> int:
        try:
            return os.path.getsize(path)
        except Exception:
            return 0

    @staticmethod
    def _load_image_rgb(path: str) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    @staticmethod
    def _get_image_meta(path: str) -> ImageMeta:
        size_bytes = ImageDeduplication._safe_stat_size(path)
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                w, h = im.size
                fmt = (im.format or "").upper()
        except Exception:
            w, h, fmt = 0, 0, ""
        return ImageMeta(path=path, size_bytes=size_bytes, width=w, height=h, fmt=fmt)

    @staticmethod
    def _canonical_score(meta: ImageMeta) -> Tuple[int, int]:
        return (meta.width * meta.height, meta.size_bytes)

    @staticmethod
    def _choose_keep(paths: List[str]) -> str:
        metas = [ImageDeduplication._get_image_meta(p) for p in paths]
        metas.sort(key=ImageDeduplication._canonical_score, reverse=True)
        return metas[0].path if metas else paths[0]

    @staticmethod
    def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> Tuple[str, int]:
        h = hashlib.sha256()
        read = 0
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
                read += len(b)
        return h.hexdigest(), read

    @staticmethod
    def _dhash64(img: Image.Image) -> int:
        g = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        px = np.asarray(g, dtype=np.int16)
        diff = px[:, 1:] > px[:, :-1]
        bits = diff.flatten()
        val = 0
        for i, bit in enumerate(bits):
            if bit:
                val |= (1 << i)
        return val

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def _write_report_jsonl(
        self,
        report_path: str,
        groups: List[List[str]],
        method: str,
        extra: Optional[Dict] = None
    ):
        self._ensure_dir(os.path.dirname(report_path))
        extra = extra or {}
        with open(report_path, "w", encoding="utf-8") as f:
            for gid, paths in enumerate(groups):
                keep = self._choose_keep(paths)
                dups = [p for p in paths if p != keep]
                f.write(json.dumps({
                    "group_id": gid,
                    "method": method,
                    "keep": keep,
                    "duplicates": dups,
                    "all": paths,
                    **extra
                }, ensure_ascii=False) + "\n")

    def _move_duplicates(
        self,
        groups: List[List[str]],
        duplicates_dir: str,
        update_status: Callable[[str], None] = default_update_status,
    ):
        self._ensure_dir(duplicates_dir)
        moved = 0

        for paths in groups:
            keep = self._choose_keep(paths)
            for p in paths:
                if p == keep:
                    continue
                try:
                    base = os.path.basename(p)
                    dest = os.path.join(duplicates_dir, base)

                    if os.path.exists(dest):
                        stem, ext = os.path.splitext(base)
                        k = 1
                        while True:
                            dest = os.path.join(duplicates_dir, f"{stem}__dup{k}{ext}")
                            if not os.path.exists(dest):
                                break
                            k += 1

                    os.replace(p, dest)
                    moved += 1
                except Exception:
                    logging.error(f"Failed to move duplicate: {p}", exc_info=True)

        update_status(f"Moved {moved} duplicate files to: {duplicates_dir}")

    def copy_unique_to_output(
        self,
        all_paths: List[str],
        groups: List[List[str]],
        output_dir: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        randomize: bool = False,
    ):
        """
        Copy unique images (non-duplicates + best from each duplicate group) to output_dir.
        """
        import shutil
        import random
        self._ensure_dir(output_dir)

        # Build set of all duplicate paths (excluding the "keep" from each group)
        duplicates_set = set()
        keep_set = set()
        for paths in groups:
            keep = self._choose_keep(paths)
            keep_set.add(keep)
            for p in paths:
                if p != keep:
                    duplicates_set.add(p)

        # Unique images = all images that are NOT duplicates
        # This includes: images not in any group + the "keep" image from each group
        unique_paths = [p for p in all_paths if p not in duplicates_set]
        
        # Randomize order if requested
        if randomize:
            random.shuffle(unique_paths)
            update_status("Shuffled output order")

        total = len(unique_paths)
        dup_count = len(duplicates_set)
        update_status(f"Found {len(groups)} duplicate groups ({dup_count} files). Copying {total} unique images...")
        copied = 0

        for i, p in enumerate(unique_paths):
            try:
                base = os.path.basename(p)
                dest = os.path.join(output_dir, base)

                # Handle filename collisions
                if os.path.exists(dest):
                    stem, ext = os.path.splitext(base)
                    k = 1
                    while True:
                        dest = os.path.join(output_dir, f"{stem}_{k}{ext}")
                        if not os.path.exists(dest):
                            break
                        k += 1

                shutil.copy2(p, dest)
                copied += 1

                if i % 50 == 0:
                    update_progress(i, total)
            except Exception:
                logging.error(f"Failed to copy: {p}", exc_info=True)

        update_progress(total, total)
        update_status(f"Copied {copied} unique images to: {output_dir}")

    # ✅ FIXED: robust total bytes estimate + failsafe growth
    def perform_sha256_image_dedup(
        self,
        inputs: List[str],
        report_path: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        move_duplicates: bool = False,
        duplicates_dir: Optional[str] = None,
        precomputed_paths: Optional[List[str]] = None,
    ) -> List[List[str]]:
        try:
            paths = precomputed_paths if precomputed_paths else self.iter_image_paths(inputs)
            if not paths:
                update_status("No images found.")
                return []

            sizes = []
            unknown = 0
            for p in paths:
                s = self._safe_stat_size(p)
                if s > 0:
                    sizes.append(s)
                else:
                    unknown += 1

            if sizes:
                sizes_sorted = sorted(sizes)
                median = sizes_sorted[len(sizes_sorted) // 2]
                total_bytes = sum(sizes) + unknown * max(median, 1)
            else:
                total_bytes = max(len(paths), 1)

            update_status(f"SHA-256 scanning {len(paths)} images...")
            bytes_done = 0
            groups_by_hash: Dict[str, List[str]] = {}
            completed = 0

            # Parallel SHA-256 hashing
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self._sha256_file, p): p for p in paths}
                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        digest, read = future.result()
                        bytes_done += read
                        if bytes_done > total_bytes:
                            total_bytes = bytes_done + 1
                        groups_by_hash.setdefault(digest, []).append(p)
                    except Exception:
                        logging.error(f"SHA-256 failed for: {p}", exc_info=True)

                    completed += 1
                    if completed % 50 == 0:
                        update_progress(bytes_done, total_bytes)

            update_progress(total_bytes, total_bytes)

            dup_groups = [g for g in groups_by_hash.values() if len(g) > 1]
            self._write_report_jsonl(report_path, dup_groups, method="sha256_exact")

            update_status(f"SHA-256 complete. Duplicate groups: {len(dup_groups)}. Report: {report_path}")

            if move_duplicates and duplicates_dir:
                self._move_duplicates(dup_groups, duplicates_dir, update_status=update_status)

            return dup_groups
        except Exception as e:
            logging.error(f"Error during SHA-256 image dedup: {e}", exc_info=True)
            update_status(f"SHA-256 failed: {type(e).__name__}: {e}")
            return []

    # (rest unchanged)
    def perform_dhash_dedup(
        self,
        inputs: List[str],
        report_path: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        move_duplicates: bool = False,
        duplicates_dir: Optional[str] = None,
        precomputed_paths: Optional[List[str]] = None,
    ) -> List[List[str]]:
        try:
            paths = precomputed_paths if precomputed_paths else self.iter_image_paths(inputs)
            n = len(paths)
            if n == 0:
                update_status("No images found.")
                return []

            update_status(f"dHash scanning {n} images...")
            update_progress(0, n)

            items: List[Tuple[str, int, ImageMeta]] = []

            def compute_dhash(p: str) -> Optional[Tuple[str, int, ImageMeta]]:
                try:
                    img = self._load_image_rgb(p)
                    dh = self._dhash64(img)
                    meta = self._get_image_meta(p)
                    return (p, dh, meta)
                except Exception:
                    return None

            # Parallel dHash computation
            completed = 0
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(compute_dhash, p) for p in paths]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        items.append(result)
                    completed += 1
                    if completed % 100 == 0:
                        update_progress(completed, n)

            update_progress(n, n)

            if not items:
                update_status("No readable images found for dHash.")
                return []

            PREFIX = int(self.prefix_bits)
            PREFIX = max(1, min(PREFIX, 32))
            prefix_mask = (1 << PREFIX) - 1

            buckets: Dict[int, List[int]] = {}
            for idx, (_, dh, _) in enumerate(items):
                key = (dh >> (64 - PREFIX)) & prefix_mask
                buckets.setdefault(key, []).append(idx)

            visited = set()
            groups: List[List[str]] = []

            for _, idxs in buckets.items():
                for pos, i in enumerate(idxs):
                    if i in visited:
                        continue
                    _, base_hash, _ = items[i]
                    group = [i]

                    for j in idxs[pos + 1:]:
                        if j in visited:
                            continue
                        _, h2, _ = items[j]
                        if self._hamming(base_hash, h2) <= int(self.dhash_hamming_threshold):
                            group.append(j)

                    for k in group:
                        visited.add(k)

                    if len(group) > 1:
                        groups.append([items[k][0] for k in group])

            self._write_report_jsonl(
                report_path,
                groups,
                method=f"dhash64_h{int(self.dhash_hamming_threshold)}_p{PREFIX}",
                extra={"hamming_threshold": int(self.dhash_hamming_threshold), "prefix_bits": PREFIX}
            )

            update_status(f"dHash complete. Duplicate groups: {len(groups)}. Report: {report_path}")

            if move_duplicates and duplicates_dir:
                self._move_duplicates(groups, duplicates_dir, update_status=update_status)

            return groups
        except Exception as e:
            logging.error(f"Error during dHash dedup: {e}", exc_info=True)
            update_status(f"dHash failed: {type(e).__name__}: {e}")
            return []

    def perform_embedding_dedup(
        self,
        inputs: List[str],
        report_path: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        top_k: int = 30,
        move_duplicates: bool = False,
        duplicates_dir: Optional[str] = None,
        precomputed_paths: Optional[List[str]] = None,
    ) -> List[List[str]]:
        try:
            paths = precomputed_paths if precomputed_paths else self.iter_image_paths(inputs)
            n = len(paths)
            if n == 0:
                update_status("No images found.")
                return []

            total_steps = 2 * n
            update_status(f"Embedding scanning {n} images (model: {self.embed_model_name})...")
            update_progress(0, total_steps)

            with suppress_stdout_stderr():
                import torch
                from transformers import CLIPImageProcessor, CLIPModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            use_fp16 = device == "cuda"
            update_status(f"Loading CLIP model on {device}{'(FP16)' if use_fp16 else ''}...")
            with suppress_stdout_stderr():
                model = CLIPModel.from_pretrained(
                    self.embed_model_name,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if use_fp16 else torch.float32,
                )
                processor = CLIPImageProcessor.from_pretrained(self.embed_model_name)
            model = model.to(device)  # type: ignore[call-arg]
            model.eval()
            update_status(f"Embedding {n} images on {device}...")

            good_paths: List[str] = []
            embs: List[np.ndarray] = []
            batch_size = 64 if device == "cuda" else 8

            # Helper for parallel image loading
            def load_image_safe(p: str) -> Optional[Tuple[str, Image.Image]]:
                try:
                    return (p, self._load_image_rgb(p))
                except Exception:
                    return None

            for i in range(0, n, batch_size):
                batch_paths = paths[i:i + batch_size]

                # Parallel image loading
                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(load_image_safe, batch_paths))

                imgs = []
                local_paths = []
                for r in results:
                    if r:
                        local_paths.append(r[0])
                        imgs.append(r[1])

                if imgs:
                    inputs_t = processor(images=imgs, return_tensors="pt")
                    pixel_values = inputs_t["pixel_values"].to(device, dtype=torch.float16 if use_fp16 else torch.float32)
                    with torch.inference_mode():
                        feats = model.get_image_features(pixel_values=pixel_values)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    embs.append(feats.detach().cpu().float().numpy().astype("float32"))
                    good_paths.extend(local_paths)

                update_progress(min(i + batch_size, n), total_steps)

            if not embs:
                update_status("No readable images for embeddings.")
                return []

            E = np.ascontiguousarray(np.vstack(embs), dtype=np.float32)
            m = E.shape[0]

            groups: List[List[str]] = []
            visited = set()

            try:
                import faiss
                index = faiss.IndexFlatIP(E.shape[1])
                index.add(E)  # type: ignore[call-arg]
                D, I = index.search(E, int(top_k) + 1)  # type: ignore[call-arg]
            except Exception:
                D = np.full((m, int(top_k) + 1), -1.0, dtype=np.float32)
                I = np.full((m, int(top_k) + 1), -1, dtype=np.int32)
                for i in range(m):
                    sims = E @ E[i]
                    k = min(int(top_k) + 1, m)
                    idxs = np.argpartition(-sims, k - 1)[:k]
                    idxs = idxs[np.argsort(-sims[idxs])]
                    I[i, :k] = idxs
                    D[i, :k] = sims[idxs]

            processed = 0
            thr = float(self.embedding_threshold)

            for i in range(m):
                if i in visited:
                    processed += 1
                    if processed % 100 == 0:
                        update_progress(n + processed, total_steps)
                    continue

                group = [i]
                for j, sim in zip(I[i][1:], D[i][1:]):
                    if j < 0 or j in visited:
                        continue
                    if float(sim) >= thr:
                        group.append(int(j))

                for k in group:
                    visited.add(k)

                if len(group) > 1:
                    groups.append([good_paths[k] for k in group])

                processed += 1
                if processed % 100 == 0:
                    update_progress(n + processed, total_steps)

            update_progress(total_steps, total_steps)

            self._write_report_jsonl(
                report_path,
                groups,
                method=f"clip_cos{thr}_k{int(top_k)}",
                extra={"cosine_threshold": thr, "top_k": int(top_k), "device": device}
            )
            update_status(f"Embedding complete. Duplicate groups: {len(groups)}. Report: {report_path}")

            if move_duplicates and duplicates_dir:
                self._move_duplicates(groups, duplicates_dir, update_status=update_status)

            return groups
        except Exception as e:
            logging.error(f"Error during embedding dedup: {e}", exc_info=True)
            update_status(f"Embedding dedup failed: {type(e).__name__}")
            return []

    def perform_text_metadata_dedup(
        self,
        input_dir: str,
        output_dir: str,
        report_path: str,
        update_status: Callable[[str], None] = default_update_status,
        update_progress: Callable[[int, int], None] = default_update_progress,
        metadata_file: str = "metadata.jsonl",
        images_subdir: str = "images",
        text_field: str = "text",
        filename_field: str = "file_name",
    ) -> Tuple[int, int]:
        """
        Deduplicate an image dataset based on text/caption similarity.
        
        Expects a folder structure like:
            input_dir/
                metadata.jsonl   (or custom name)
                images/          (or custom subdir)
                    image1.jpg
                    image2.png
                    ...
        
        Each line in metadata.jsonl should have at least:
            {"file_name": "image1.jpg", "text": "A caption describing the image"}
        
        Args:
            input_dir: Root directory containing metadata and images
            output_dir: Where to write deduplicated output
            report_path: Path to write the dedup report
            metadata_file: Name of the metadata file (default: metadata.jsonl)
            images_subdir: Name of images subdirectory (default: images)
            text_field: Field name containing the text/caption (default: text)
            filename_field: Field name containing the image filename (default: file_name)
        
        Returns:
            Tuple of (kept_count, total_count)
        """
        import shutil
        
        try:
            input_path = Path(input_dir)
            meta_path = input_path / metadata_file
            images_path = input_path / images_subdir
            
            # Also check if images are directly in input_dir (no subdir)
            if not images_path.exists():
                images_path = input_path
            
            if not meta_path.exists():
                update_status(f"❌ Metadata file not found: {meta_path}")
                return (0, 0)
            
            # Load metadata records
            update_status(f"Loading metadata from {meta_path}...")
            records = []
            with open(meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            if not records:
                update_status("❌ No valid records found in metadata.")
                return (0, 0)
            
            total = len(records)
            update_status(f"Found {total} records. Extracting text for embedding...")
            
            # Extract texts
            texts = []
            valid_records = []
            for rec in records:
                text = rec.get(text_field, "")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    valid_records.append(rec)
                else:
                    # Skip records without text
                    pass
            
            if not texts:
                update_status(f"❌ No text found in field '{text_field}'.")
                return (0, 0)
            
            update_status(f"Encoding {len(texts)} texts with {self.text_embed_model}...")
            update_progress(0, len(texts))
            
            # Load sentence transformer model
            with suppress_stdout_stderr():
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.text_embed_model)
            
            # Encode texts
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=128,
            )
            
            update_progress(len(texts) // 2, len(texts))
            
            # Deduplicate based on similarity
            update_status("Finding duplicates by text similarity...")
            kept_indices = []
            kept_embeddings = []
            duplicate_groups = []
            
            threshold = float(self.text_sim_threshold)
            
            for i, (rec, emb) in enumerate(zip(valid_records, embeddings)):
                if kept_embeddings:
                    # Compute similarity with all kept embeddings
                    sims = np.dot(kept_embeddings, emb)
                    max_sim = float(np.max(sims))
                    
                    if max_sim >= threshold:
                        # This is a duplicate - find which group it belongs to
                        max_idx = int(np.argmax(sims))
                        # Track as duplicate (for reporting)
                        continue
                
                # Keep this record
                kept_indices.append(i)
                kept_embeddings.append(emb)
                
                if i % 100 == 0:
                    update_progress(len(texts) // 2 + i // 2, len(texts))
            
            kept_records = [valid_records[i] for i in kept_indices]
            removed_count = len(valid_records) - len(kept_records)
            
            update_status(f"Keeping {len(kept_records)} / {len(valid_records)} records (removed {removed_count} duplicates)")
            
            # Create output directory
            out_path = Path(output_dir)
            out_images_path = out_path / images_subdir
            self._ensure_dir(str(out_path))
            self._ensure_dir(str(out_images_path))
            
            # Copy unique images and write new metadata
            update_status("Copying unique images to output...")
            copied = 0
            
            with open(out_path / metadata_file, 'w', encoding='utf-8') as f:
                for i, rec in enumerate(kept_records):
                    filename = rec.get(filename_field, "")
                    if filename:
                        src = images_path / filename
                        dst = out_images_path / filename
                        
                        if src.exists():
                            try:
                                shutil.copy2(str(src), str(dst))
                                copied += 1
                            except Exception:
                                logging.error(f"Failed to copy: {src}", exc_info=True)
                    
                    # Write record to new metadata
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    
                    if i % 50 == 0:
                        update_progress(len(texts) // 2 + len(texts) // 4 + i * len(texts) // (4 * len(kept_records)), len(texts))
            
            update_progress(len(texts), len(texts))
            
            # Write report
            self._ensure_dir(os.path.dirname(report_path))
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({
                    "method": "text_similarity",
                    "model": self.text_embed_model,
                    "threshold": threshold,
                    "total_records": len(valid_records),
                    "kept_records": len(kept_records),
                    "removed_duplicates": removed_count,
                    "images_copied": copied,
                    "input_dir": str(input_path),
                    "output_dir": str(out_path),
                }, ensure_ascii=False) + "\n")
            
            update_status(f"✅ Text dedup complete! Kept {len(kept_records)}/{len(valid_records)} ({copied} images copied)")
            
            return (len(kept_records), len(valid_records))
            
        except Exception as e:
            logging.error(f"Error during text metadata dedup: {e}", exc_info=True)
            update_status(f"❌ Text dedup failed: {type(e).__name__}: {e}")
            return (0, 0)
