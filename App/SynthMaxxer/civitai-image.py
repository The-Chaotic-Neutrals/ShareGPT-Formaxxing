import os
import re
import json
import time
import getpass
import threading
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_BASE = "https://civitai.com/api/v1/images"
LOG_PATH = "downloaded_urls.log"
TAG_RE = re.compile(r"<.*?>", re.DOTALL)

API_KEY_PATH = ".civitai_api_key"

# Fixed defaults
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

API_LIMIT = 200
MAX_EMPTY_BATCHES = 40

# Hard wall-clock limit for the API paging request (prevents "hang forever")
API_WATCHDOG_SECONDS = 90

# Metadata JSONL (one record per successfully downloaded image)
META_JSONL_PATH = "image_metadata.jsonl"


# -----------------------------
# API key persistence
# -----------------------------
def load_saved_api_key(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        return key if key else None
    except Exception:
        return None


def save_api_key(path: str, key: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(key.strip())
    except Exception:
        pass


# -----------------------------
# CLI helpers
# -----------------------------
def ask_int(prompt, default=None, min_value=None):
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not s and default is not None:
            v = default
        else:
            try:
                v = int(s)
            except ValueError:
                print("Enter an integer.")
                continue
        if min_value is not None and v < min_value:
            print(f"Must be >= {min_value}")
            continue
        return v


def ask_str(prompt, default=""):
    s = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return s if s else default


def ask_yn(prompt, default=False):
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} ({d}): ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def parse_csv(s):
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def ask_sort_mode():
    print("\n--- Sort mode ---")
    print("1) Newest (cursor paging, recommended)")
    print("2) Most Reactions (page paging)")
    print("3) Most Comments (page paging)")
    choice = input("Choose sort [1-3]: ").strip() or "1"
    if choice == "2":
        return "Most Reactions"
    if choice == "3":
        return "Most Comments"
    return "Newest"


def ask_nsfw_level():
    """
    Returns:
      None        -> don't send nsfw param (Any)
      "None"/"Soft"/"Mature"/"X" -> send nsfw param with that value
    """
    print("\n--- NSFW filter ---")
    print("1) Any (no filter)")
    print("2) None (SFW only)")
    print("3) Soft")
    print("4) Mature")
    print("5) X (explicit)")
    choice = input("Choose [1-5]: ").strip() or "1"
    mapping = {
        "1": None,
        "2": "None",
        "3": "Soft",
        "4": "Mature",
        "5": "X",
    }
    if choice not in mapping:
        print("Invalid choice; using Any.")
        return None
    return mapping[choice]


# -----------------------------
# HTTP session + retries
# -----------------------------
def make_session():
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("https://", adapter)
    return session


def sleep_backoff(i: int, base: float = 1.0, cap: float = 20.0):
    time.sleep(min(base * (2**i), cap))


def get_with_watchdog(session, url, *, headers, params, timeout, watchdog_seconds=90):
    """
    Runs requests.get in a background thread and enforces a hard wall-clock timeout.
    Windows-safe. If the request hangs, we bail instead of freezing forever.
    """
    result = {"resp": None, "err": None}

    def _run():
        try:
            result["resp"] = session.get(url, headers=headers, params=params, timeout=timeout)
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(watchdog_seconds)

    if t.is_alive():
        raise TimeoutError(f"Hard timeout: request exceeded {watchdog_seconds}s")

    if result["err"] is not None:
        raise result["err"]

    return result["resp"]


# -----------------------------
# Persistence
# -----------------------------
def load_downloaded():
    if not os.path.exists(LOG_PATH):
        return set()
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def mark_downloaded(url):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def append_metadata_jsonl(path: str, obj: dict):
    try:
        line = json.dumps(obj, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# -----------------------------
# Prompt extraction + matching
# -----------------------------
def extract_prompt_text(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    for k in (
        "prompt",
        "Prompt",
        "positivePrompt",
        "positive_prompt",
        "positive",
        "Positive prompt",
        "caption",
        "description",
    ):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    try:
        return json.dumps(meta, ensure_ascii=False)
    except Exception:
        return ""


def normalize_text(s: str) -> str:
    s = TAG_RE.sub(" ", s)
    return s.lower()


def text_pass(text: str, include_terms: list[str], exclude_terms: list[str]) -> bool:
    """
    INCLUDE: OR semantics (if include_terms is non-empty, at least one must match)
    EXCLUDE: OR semantics (if any exclude term matches, reject)
    """
    t = normalize_text(text)

    # INCLUDE: OR logic
    if include_terms and not any(x in t for x in include_terms):
        return False

    # EXCLUDE: OR logic
    for x in exclude_terms:
        if x in t:
            return False

    return True


# -----------------------------
# Download/save
# -----------------------------
def guess_ext(url):
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".jpg"


def download_bytes(session: requests.Session, url: str) -> bytes | None:
    for i in range(5):
        try:
            r = session.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            if i == 4:
                return None
            sleep_backoff(i)
            continue

        if r.status_code < 400:
            return r.content

        if r.status_code in (429, 500, 502, 503, 504):
            if i == 4:
                return None
            sleep_backoff(i)
            continue

        return None
    return None


def save_image_and_text(raw: bytes, url: str, image_id: int, out_dir: str, prompt_text: str):
    ext = guess_ext(url)
    img_path = os.path.join(out_dir, f"{image_id}{ext}")
    with open(img_path, "wb") as f:
        f.write(raw)

    txt_path = os.path.join(out_dir, f"{image_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)


# -----------------------------
# Main
# -----------------------------
def main():
    print("=== Civitai Image Downloader (meta prompt match, sequential downloads) ===")

    saved_key = load_saved_api_key(API_KEY_PATH)
    if saved_key:
        api_key = getpass.getpass("API key (hidden) [Enter to reuse saved]: ").strip()
        if not api_key:
            api_key = saved_key
    else:
        api_key = getpass.getpass("API key (hidden): ").strip()

    if api_key:
        save_api_key(API_KEY_PATH, api_key)

    # Force connection close to avoid stuck keep-alive sockets
    headers = {"Connection": "close"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    out_dir = ask_str("Download folder", "civitai_images")
    os.makedirs(out_dir, exist_ok=True)

    max_images = ask_int("Max images", 100, 1)
    min_w = ask_int("Min width", 0, 0)
    min_h = ask_int("Min height", 0, 0)

    nsfw_level = ask_nsfw_level()

    # Clarified text so it doesn't sound like a preference question
    include_terms = parse_csv(
        ask_str("Include terms (comma-separated; ANY match passes; blank = no include filter)", "")
    )
    exclude_terms = parse_csv(
        ask_str("Exclude terms (comma-separated; ANY match blocks; blank = no exclude filter)", "")
    )

    save_meta_jsonl = ask_yn(f"Save metadata JSONL ({META_JSONL_PATH})", True)

    has_search_terms = bool(include_terms or exclude_terms)

    sort_mode = ask_sort_mode()

    session = make_session()
    downloaded = load_downloaded()

    base_params = {
        "limit": API_LIMIT,
        "sort": sort_mode,
        "period": "AllTime",
        "withMeta": "true",
        "include": ["metaSelect", "tagIds", "profilePictures"],
    }

    if nsfw_level is not None:
        base_params["nsfw"] = nsfw_level

    def meta_payload(img):
        return {
            "ts": time.time(),
            "id": img.get("id"),
            "url": img.get("url"),
            "width": img.get("width"),
            "height": img.get("height"),
            "nsfw": img.get("nsfw"),
            "nsfwLevel": img.get("nsfwLevel"),
            "stats": img.get("stats"),
            "meta": img.get("meta"),
            "username": img.get("username"),
            "createdAt": img.get("createdAt"),
            "postId": img.get("postId"),
            "modelVersionIds": img.get("modelVersionIds"),
        }

    saved = 0
    empty_batches = 0

    use_cursor = (sort_mode == "Newest")
    cursor = None
    page = 1

    print(f"\nStarting... (limit={API_LIMIT}, sort={sort_mode}, paging={'cursor' if use_cursor else 'page'})")

    while saved < max_images:
        params = dict(base_params)

        if use_cursor:
            if cursor:
                params["cursor"] = cursor
        else:
            params["page"] = page

        try:
            print(f"Requesting {'cursor' if use_cursor else 'page'}={cursor if use_cursor else page} ...", flush=True)
            r = get_with_watchdog(
                session,
                API_BASE,
                headers=headers,
                params=params,
                timeout=TIMEOUT,
                watchdog_seconds=API_WATCHDOG_SECONDS,
            )
            print(f"Response {r.status_code} bytes={len(r.content)}", flush=True)
        except TimeoutError as e:
            print(str(e), flush=True)
            sleep_backoff(0)
            continue
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except requests.RequestException as e:
            print(f"RequestException: {e}", flush=True)
            sleep_backoff(0)
            continue

        if r.status_code >= 400:
            if r.status_code in (429, 500, 502, 503, 504):
                sleep_backoff(0)
                continue
            print("Request failed:", r.status_code, getattr(r, "url", "<no-url>"))
            try:
                print(r.text[:500])
            except Exception:
                pass
            break

        data = r.json() if r.content else {}
        items = data.get("items") or []

        md = data.get("metadata") or {}
        if use_cursor:
            cursor = md.get("nextCursor")
        else:
            page += 1

        if not items:
            if use_cursor and cursor:
                continue
            break

        matched = []
        for img in items:
            url = img.get("url")
            if not url:
                continue

            if url in downloaded:
                continue

            if img.get("width", 0) < min_w or img.get("height", 0) < min_h:
                continue

            prompt_text = extract_prompt_text(img.get("meta") or {})
            if has_search_terms and not text_pass(prompt_text, include_terms, exclude_terms):
                continue

            matched.append((img, prompt_text))

        if not matched:
            empty_batches += 1
            print(f"Matched 0 in this batch ({empty_batches}/{MAX_EMPTY_BATCHES}). items={len(items)}", flush=True)
            if empty_batches >= MAX_EMPTY_BATCHES:
                print("Stopping: too many empty batches.")
                break
            continue
        else:
            empty_batches = 0

        remaining = max_images - saved
        take = min(len(matched), remaining)

        for img, prompt_text in matched[:take]:
            if saved >= max_images:
                break

            url = img["url"]
            image_id = img["id"]

            raw = download_bytes(session, url)
            if not raw:
                continue

            save_image_and_text(raw, url, image_id, out_dir, prompt_text)

            mark_downloaded(url)
            downloaded.add(url)
            saved += 1

            if save_meta_jsonl:
                append_metadata_jsonl(META_JSONL_PATH, meta_payload(img))

            if saved % 25 == 0:
                print(f"Saved: {saved}/{max_images}", flush=True)

        print(f"Batch done | matched={len(matched)} | saved={saved}/{max_images}", flush=True)

    print(f"\nDone. Downloaded {saved} images.")
    print(f"Output: {os.path.abspath(out_dir)}")
    print(f"Log: {os.path.abspath(LOG_PATH)}")
    print(f"Saved key path: {os.path.abspath(API_KEY_PATH)} (add to .gitignore)")
    if save_meta_jsonl:
        print(f"Metadata JSONL: {os.path.abspath(META_JSONL_PATH)}")


if __name__ == "__main__":
    main()
