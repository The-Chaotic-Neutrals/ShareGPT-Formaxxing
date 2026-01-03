import os
import sys
import json
import hashlib
import logging
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

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
    ):
        self.dhash_hamming_threshold = dhash_hamming_threshold
        self.prefix_bits = prefix_bits
        self.embedding_threshold = embedding_threshold
        self.embed_model_name = embed_model_name

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
    ):
        """
        Copy unique images (non-duplicates + best from each duplicate group) to output_dir.
        """
        import shutil
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
