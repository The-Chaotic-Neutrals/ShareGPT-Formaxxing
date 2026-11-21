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

        def add(path):
            identity = os.path.normcase(os.path.realpath(path))
            if identity not in seen:
                seen.add(identity)
                paths.append(os.path.abspath(path))

        for raw in inputs:
            p = Path(raw)
            if p.is_dir():
                for root, directories, files in os.walk(p):
                    directories.sort()
                    for name in sorted(files):
                        if Path(name).suffix.lower() in self.IMAGE_EXTS:
                            add(Path(root) / name)
            elif p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                add(p)
        return paths

    @staticmethod
    def _write_extracted_image(output_dir, stem, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as image:
            extension = {"JPEG": "jpg", "TIFF": "tiff"}.get(
                image.format, (image.format or "png").lower()
            )
        path = Path(output_dir) / f"{stem}.{extension}"
        path.write_bytes(image_bytes)
        return str(path)

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
                        
                        out_path = self._write_extracted_image(
                            output_dir, f"parquet_{global_idx:08d}", img_bytes,
                        )

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
            from datasets import load_dataset, DatasetDict, IterableDataset, Image as HFImage
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
        
        feature = (getattr(dataset, "features", None) or {}).get(image_column)
        if isinstance(feature, HFImage):
            dataset = dataset.cast_column(image_column, HFImage(decode=False))

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
                
                image_bytes = img_data.get("bytes") if isinstance(img_data, dict) else img_data
                stem = f"hf_{idx:08d}"
                if isinstance(image_bytes, bytes):
                    out_path = self._write_extracted_image(output_dir, stem, image_bytes)
                elif hasattr(img_data, "save"):
                    out_path = os.path.join(output_dir, f"{stem}.png")
                    img_data.save(out_path)
                elif isinstance(img_data, str) and os.path.isfile(img_data):
                    out_path = self._write_extracted_image(output_dir, stem, Path(img_data).read_bytes())
                elif isinstance(img_data, dict) and img_data.get("path"):
                    # Decode path-backed Image features using the dataset library's resolver.
                    image = (
                        HFImage().decode_example(img_data, token_per_repo_id={dataset_name: token})
                        if isinstance(feature, HFImage) else None
                    )
                    if image is None:
                        continue
                    out_path = os.path.join(output_dir, f"{stem}.png")
                    with image:
                        image.save(out_path)
                else:
                    continue

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
                    **extra_meta,
                    "text": text_val,
                    "source_dataset": dataset_name,
                    "row_idx": idx,
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
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

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
                keep = paths[0]
                dups = [p for p in paths if p != keep]
                f.write(json.dumps({
                    "group_id": gid,
                    "method": method,
                    "keep": keep,
                    "duplicates": dups,
                    "all": paths,
                    **extra
                }, ensure_ascii=False) + "\n")

    def kept_paths(self, all_paths, groups):
        duplicates = {path for group in groups for path in group[1:]}
        return [path for path in all_paths if path not in duplicates]

    def combine_image_groups(self, exact_groups, near_groups):
        """Expand perceptual groups after scanning only SHA-256 representatives."""
        exact_by_keep = {group[0]: group for group in exact_groups}
        combined = []
        for group in near_groups:
            combined.append([
                path for representative in group
                for path in exact_by_keep.pop(representative, [representative])
            ])
        combined.extend(exact_by_keep.values())
        return combined

    def _order_groups(self, groups):
        """Choose one representative before any file is copied or moved."""
        ordered = []
        for group in groups:
            keep = self._choose_keep(group)
            ordered.append([keep, *(p for p in group if p != keep)])
        return ordered

    def _move_duplicates(
        self,
        groups: List[List[str]],
        duplicates_dir: str,
        update_status: Callable[[str], None] = default_update_status,
        exclude=(),
    ):
        self._ensure_dir(duplicates_dir)
        moved = 0

        for paths in groups:
            keep = paths[0]
            for p in paths:
                if p == keep or p in exclude:
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

                    shutil.move(p, dest)
                    moved += 1
                except OSError as exc:
                    raise OSError(f"Could not move duplicate {p}: {exc}") from exc

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

        unique_paths = self.kept_paths(all_paths, groups)

        # Randomize order if requested
        if randomize:
            random.shuffle(unique_paths)
            update_status("Shuffled output order")

        total = len(unique_paths)
        dup_count = len(all_paths) - total
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
            except OSError as exc:
                raise OSError(f"Could not copy retained image {p}: {exc}") from exc

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
        annotation_keys=None,
    ) -> List[List[str]]:
        annotation_keys = annotation_keys or {}
        try:
            paths = precomputed_paths if precomputed_paths is not None else self.iter_image_paths(inputs)
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
            groups_by_hash: Dict[tuple, List[str]] = {}
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
                        groups_by_hash.setdefault((digest, annotation_keys.get(p)), []).append(p)
                    except Exception:
                        logging.error(f"SHA-256 failed for: {p}", exc_info=True)

                    completed += 1
                    if completed % 50 == 0:
                        update_progress(bytes_done, total_bytes)

            update_progress(total_bytes, total_bytes)

            order = {path: index for index, path in enumerate(paths)}
            dup_groups = self._order_groups([
                sorted(group, key=order.__getitem__)
                for group in groups_by_hash.values() if len(group) > 1
            ])
            dup_groups.sort(key=lambda group: order[group[0]])
            self._write_report_jsonl(report_path, dup_groups, method="sha256_exact")

            update_status(f"SHA-256 complete. Duplicate groups: {len(dup_groups)}. Report: {report_path}")

            if move_duplicates and duplicates_dir:
                self._move_duplicates(dup_groups, duplicates_dir, update_status=update_status)

            return dup_groups
        except Exception as e:
            logging.error(f"Error during SHA-256 image dedup: {e}", exc_info=True)
            update_status(f"SHA-256 failed: {type(e).__name__}: {e}")
            raise

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
        annotation_keys=None,
    ) -> List[List[str]]:
        annotation_keys = annotation_keys or {}
        try:
            paths = precomputed_paths if precomputed_paths is not None else self.iter_image_paths(inputs)
            n = len(paths)
            if n == 0:
                update_status("No images found.")
                return []

            update_status(f"dHash scanning {n} images...")
            update_progress(0, n)

            items = []

            def compute_dhash(p: str):
                try:
                    with self._load_image_rgb(p) as img:
                        dhash = self._dhash64(img)
                        color = np.asarray(img.resize((8, 8), Image.Resampling.BOX), dtype=np.int16)
                    return p, dhash, color, self._get_image_meta(p)
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

            # Compare to the same best-quality image that will represent the group.
            order = {path: index for index, path in enumerate(paths)}
            items.sort(key=lambda item: (*self._canonical_score(item[3]), -order[item[0]]), reverse=True)
            buckets = {}
            for idx, (path, dh, _, _) in enumerate(items):
                key = ((dh >> (64 - PREFIX)) & prefix_mask, annotation_keys.get(path))
                buckets.setdefault(key, []).append(idx)

            visited = set()
            groups: List[List[str]] = []

            for _, idxs in buckets.items():
                for pos, i in enumerate(idxs):
                    if i in visited:
                        continue
                    _, base_hash, base_color, _ = items[i]
                    group = [i]

                    for j in idxs[pos + 1:]:
                        if j in visited:
                            continue
                        _, candidate_hash, candidate_color, _ = items[j]
                        # Grayscale-gradient hashes alone collapse unrelated flat-color images.
                        if (
                            self._hamming(base_hash, candidate_hash) <= int(self.dhash_hamming_threshold)
                            and np.abs(base_color - candidate_color).mean() <= 24
                        ):
                            group.append(j)

                    for k in group:
                        visited.add(k)

                    if len(group) > 1:
                        groups.append([items[k][0] for k in group])

            groups = self._order_groups(groups)
            self._write_report_jsonl(
                report_path,
                groups,
                method=f"dhash64_h{int(self.dhash_hamming_threshold)}_p{PREFIX}",
                extra={"hamming_threshold": int(self.dhash_hamming_threshold), "prefix_bits": PREFIX, "color_mean_difference_limit": 24}
            )

            update_status(f"dHash complete. Duplicate groups: {len(groups)}. Report: {report_path}")

            if move_duplicates and duplicates_dir:
                self._move_duplicates(groups, duplicates_dir, update_status=update_status)

            return groups
        except Exception as e:
            logging.error(f"Error during dHash dedup: {e}", exc_info=True)
            update_status(f"dHash failed: {type(e).__name__}: {e}")
            raise

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
        annotation_keys=None,
    ) -> List[List[str]]:
        annotation_keys = annotation_keys or {}
        try:
            paths = precomputed_paths if precomputed_paths is not None else self.iter_image_paths(inputs)
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
                        features = model.get_image_features(pixel_values=pixel_values)
                        feats = features if torch.is_tensor(features) else features.pooler_output
                        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
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
                for j, sim in zip(I[i], D[i]):
                    if j < 0 or j == i or j in visited:
                        continue
                    if float(sim) >= thr and annotation_keys.get(good_paths[i]) == annotation_keys.get(good_paths[j]):
                        group.append(int(j))

                for k in group:
                    visited.add(k)

                if len(group) > 1:
                    groups.append([good_paths[k] for k in group])

                processed += 1
                if processed % 100 == 0:
                    update_progress(n + processed, total_steps)

            update_progress(total_steps, total_steps)

            groups = self._order_groups(groups)
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
            raise

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
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            
            # Load metadata records
            update_status(f"Loading metadata from {meta_path}...")
            records = []
            with open(meta_path, 'r', encoding='utf-8') as f:
                for number, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{meta_path}, line {number}: {exc.msg}") from exc
                        if not isinstance(record, dict):
                            raise ValueError(f"{meta_path}, line {number}: expected a JSON object.")
                        records.append(record)
            
            if not records:
                update_status("❌ No valid records found in metadata.")
                return (0, 0)
            
            total = len(records)
            update_status(f"Found {total} records. Extracting text for embedding...")
            
            indexed_texts = [
                (index, text.strip()) for index, record in enumerate(records)
                if isinstance(text := record.get(text_field), str) and text.strip()
            ]
            dropped = set()
            if indexed_texts:
                with suppress_stdout_stderr():
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(self.text_embed_model)
                embeddings = model.encode(
                    [text for _, text in indexed_texts], normalize_embeddings=True,
                    show_progress_bar=False, batch_size=128,
                )
                kept_embeddings = []
                for (index, _), embedding in zip(indexed_texts, embeddings):
                    if kept_embeddings and float(np.max(np.dot(kept_embeddings, embedding))) >= self.text_sim_threshold:
                        dropped.add(index)
                    else:
                        kept_embeddings.append(embedding)

            kept_records = [record for index, record in enumerate(records) if index not in dropped]
            removed_count = len(dropped)
            update_status(f"Keeping {len(kept_records)} / {total} records (removed {removed_count} duplicates).")

            # Create output directory
            out_path = Path(output_dir)
            out_images_path = out_path / images_subdir
            self._ensure_dir(str(out_path))
            self._ensure_dir(str(out_images_path))
            
            # Copy unique images and write new metadata
            update_status("Copying unique images to output...")
            copied = 0
            
            source_root = images_path.resolve()
            destination_root = out_images_path.resolve()
            if destination_root == source_root or (out_path / metadata_file).resolve() == meta_path.resolve():
                raise ValueError("Choose an output folder separate from the input dataset.")

            copies = []
            for record in kept_records:
                filename = record.get(filename_field)
                if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
                    raise ValueError(f"Expected a relative image filename, got {filename!r}.")
                src = (source_root / filename).resolve()
                dst = (destination_root / filename).resolve()
                if not src.is_relative_to(source_root) or not dst.is_relative_to(destination_root):
                    raise ValueError(f"Image filename escapes its dataset folder: {filename!r}.")
                if not src.is_file():
                    raise FileNotFoundError(f"Image not found for metadata record: {filename!r}.")
                copies.append((src, dst))
            for src, dst in copies:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            with open(out_path / metadata_file, "w", encoding="utf-8") as f:
                for record in kept_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            update_progress(total, total)

            # Write report
            self._ensure_dir(os.path.dirname(report_path))
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({
                    "method": "text_similarity",
                    "model": self.text_embed_model,
                    "threshold": self.text_sim_threshold,
                    "total_records": total,
                    "kept_records": len(kept_records),
                    "removed_duplicates": removed_count,
                    "images_copied": copied,
                    "input_dir": str(input_path),
                    "output_dir": str(out_path),
                }, ensure_ascii=False) + "\n")
            
            update_status(f"✅ Text dedup complete! Kept {len(kept_records)}/{total} ({copied} images copied)")
            
            return (len(kept_records), total)
            
        except Exception as e:
            logging.error(f"Error during text metadata dedup: {e}", exc_info=True)
            update_status(f"❌ Text dedup failed: {type(e).__name__}: {e}")
            raise
