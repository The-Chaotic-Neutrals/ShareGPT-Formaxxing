import json
import os
import logging
import queue
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from safetensors import SafetensorError

# Silence TorchDynamo warnings
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

MODEL_SPEC = {
    "model_name": "garak-llm/garak-refusal-detector",
    "refusal_index": 0,
    "split_strategy": "entry_then_sentence_long",
    "max_tokens": 8192,
}

# Global state
nlp = None
tokenizers = {}
producer_tokenizers = {}
models = {}
resolved_label_info = {}
inference_precision = "fp16"

total_input_count = 0
total_kept_count = 0
total_refusal_count = 0
total_clean_count = 0


class ProcessingCancelled(Exception):
    pass


def get_model_name():
    return MODEL_SPEC["model_name"]


def get_refusal_index():
    model_name = get_model_name()
    info = resolved_label_info.get(model_name)
    if info and info.get("refusal_id") is not None:
        return int(info["refusal_id"])
    return MODEL_SPEC["refusal_index"]


def _normalize_label_text(label):
    return str(label).strip().lower().replace("_", "-")


def _is_refusal_label(label):
    text = _normalize_label_text(label)
    return ("refus" in text) and ("non-refus" not in text)


def _is_non_refusal_label(label):
    text = _normalize_label_text(label)
    return (
        ("non-refus" in text)
        or ("compliance" in text)
        or ("safe" in text)
        or ("clean" in text)
        or (text == "comply")
    )


def get_split_strategy():
    return MODEL_SPEC.get("split_strategy", "sentence")


def get_split_strategy_label():
    strategy = get_split_strategy()
    if strategy == "entry_then_sentence_long":
        return "entry + sentence fallback for long inputs"
    if strategy == "entry_pair":
        return "prompt/response paired entry"
    return strategy


def get_model_max_tokens():
    return int(MODEL_SPEC.get("max_tokens", 512))


def _resolve_label_info(model, fallback_refusal_index):
    config = model.config
    id_to_label = {}

    label_mapping = getattr(config, "label_mapping", None)
    if isinstance(label_mapping, dict):
        for k, v in label_mapping.items():
            try:
                id_to_label[int(k)] = _normalize_label_text(v)
            except (TypeError, ValueError):
                continue

    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict):
        for k, v in id2label.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if idx not in id_to_label:
                id_to_label[idx] = _normalize_label_text(v)

    label2id = getattr(config, "label2id", None)
    if isinstance(label2id, dict):
        for k, v in label2id.items():
            try:
                idx = int(v)
            except (TypeError, ValueError):
                continue
            if idx not in id_to_label:
                id_to_label[idx] = _normalize_label_text(k)

    refusal_id = None
    non_refusal_id = None
    for idx, label in id_to_label.items():
        if refusal_id is None and _is_refusal_label(label):
            refusal_id = idx
        if non_refusal_id is None and _is_non_refusal_label(label):
            non_refusal_id = idx

    if refusal_id is None:
        refusal_id = int(fallback_refusal_index)

    if non_refusal_id is None and isinstance(getattr(config, "num_labels", None), int) and config.num_labels == 2:
        non_refusal_id = 1 - int(refusal_id)

    return {
        "refusal_id": int(refusal_id),
        "non_refusal_id": None if non_refusal_id is None else int(non_refusal_id),
        "id_to_label": id_to_label,
    }


def initialize_models(status_update_callback=None):
    global nlp, tokenizers, producer_tokenizers, models

    # Force device to GPU if available, no CPU fallback or ONNX
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enable flash attention if supported (faster inference)
    if device == 'cuda' and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            if status_update_callback:
                status_update_callback("Flash attention enabled for faster inference.")
        except Exception:
            if status_update_callback:
                status_update_callback("Flash attention not available, using standard attention.")

    model_name = get_model_name()

    if model_name not in tokenizers:
        if status_update_callback:
            status_update_callback("Loading tokenizer...")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    if model_name not in models:
        if status_update_callback:
            status_update_callback("Loading model weights...")
        try:
            models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(torch.device(device))
        except (SafetensorError, OSError, ValueError) as e:
            err = str(e)
            if "InvalidHeaderDeserialization" in err or "Error while deserializing header" in err:
                if status_update_callback:
                    status_update_callback("Detected corrupted local safetensors cache, re-downloading model files...")
                if model_name in tokenizers:
                    del tokenizers[model_name]
                tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    force_download=True,
                )
                if model_name in producer_tokenizers:
                    del producer_tokenizers[model_name]
                producer_tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    force_download=True,
                    use_fast=False,
                )
                models[model_name] = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    force_download=True,
                ).eval().to(torch.device(device))
            else:
                raise

    if model_name not in resolved_label_info:
        fallback = MODEL_SPEC["refusal_index"]
        resolved_label_info[model_name] = _resolve_label_info(models[model_name], fallback)

    if status_update_callback:
        info = resolved_label_info.get(model_name, {})
        refusal_id = info.get("refusal_id")
        non_refusal_id = info.get("non_refusal_id")
        status_update_callback(
            f"Resolved labels: refusal_id={refusal_id}, non_refusal_id={non_refusal_id}"
        )


def update_device_preference(gpu_var, status_update_callback=None):
    if status_update_callback:
        status_update_callback("Using GPU only (CPU/ONNX support removed)")
    initialize_models(status_update_callback=status_update_callback)


def filter_conversations(input_file_entry, threshold_entry, batch_size_entry,
                         conversation_batch_size_entry=None, precision_entry=None,
                         split_tokens_entry=None,
                         status_update_callback=None, counts_update_callback=None,
                         progress_update_callback=None,
                         stop_requested_callback=None):
    input_file = input_file_entry.get()
    if not input_file.endswith('.jsonl'):
        if status_update_callback:
            status_update_callback("Invalid file type. Please select a .jsonl file.")
        return

    try:
        threshold = float(threshold_entry.get())
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0.")
    except ValueError as e:
        if status_update_callback:
            status_update_callback(f"Error: {e}")
        return

    try:
        batch_size = int(batch_size_entry.get())
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
    except ValueError as e:
        if status_update_callback:
            status_update_callback(f"Error: {e}")
        return

    try:
        if conversation_batch_size_entry is None:
            conversation_batch_size = batch_size
        else:
            conversation_batch_size = int(conversation_batch_size_entry.get())
        if conversation_batch_size <= 0:
            raise ValueError("Conversation batch size must be a positive integer.")
    except ValueError as e:
        if status_update_callback:
            status_update_callback(f"Error: {e}")
        return

    precision_mode = "fp16"
    if precision_entry is not None:
        precision_mode = str(precision_entry.get()).strip().lower()
    if precision_mode not in {"fp16", "bf16", "fp32"}:
        if status_update_callback:
            status_update_callback("Error: Precision must be one of fp16, bf16, fp32")
        return

    try:
        backend_cap = get_model_max_tokens()
        if split_tokens_entry is None:
            split_token_limit = backend_cap
        else:
            split_token_limit = int(split_tokens_entry.get())
        split_token_limit = max(32, min(split_token_limit, backend_cap))
    except ValueError as e:
        if status_update_callback:
            status_update_callback(f"Error: {e}")
        return

    # Default to outputs folder in repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    output_dir = os.path.join(repo_root, "Outputs")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

    run_filter_streaming(
        input_file,
        output_file,
        threshold,
        batch_size,
        conversation_batch_size,
        precision_mode,
        split_token_limit,
        status_update_callback,
        counts_update_callback,
        progress_update_callback,
        stop_requested_callback,
    )


def run_filter_streaming(input_file, output_file, threshold, batch_size,
                         conversation_batch_size, precision_mode, split_token_limit,
                         status_update_callback=None, counts_update_callback=None,
                         progress_update_callback=None,
                         stop_requested_callback=None):
    try:
        # Get total lines for progress tracking
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = open(output_file, 'w', encoding='utf-8')

        global total_refusal_count, total_clean_count, total_input_count, total_kept_count
        total_refusal_count = 0
        total_clean_count = 0
        total_input_count = 0
        total_kept_count = 0
        if status_update_callback:
            status_update_callback(f"Streaming and filtering started... Total lines: {total_lines}")

        queue_depth = max(4, min(max(1, conversation_batch_size), 512))
        microbatch_size = max(1, batch_size)
        if microbatch_size > 64:
            microbatch_size = 64
            if status_update_callback:
                status_update_callback("Garak Classifier microbatch capped at 64 for stability.")
        global inference_precision
        inference_precision = precision_mode
        prep_queue = queue.Queue(maxsize=queue_depth)
        sentinel = object()
        producer_state = {"input_count": 0, "entry_count": 0, "error": None}
        stop_event = threading.Event()

        if status_update_callback:
            status_update_callback(
                f"Using CPU queue depth {queue_depth}, inference microbatch {microbatch_size}"
            )
            status_update_callback(f"Inference precision: {inference_precision}")
            status_update_callback(f"Scoring strategy: {get_split_strategy_label()}")
            status_update_callback(f"Split token limit: {split_token_limit} (backend max {get_model_max_tokens()})")
            status_update_callback(f"Threshold mode: remove entries with refusal confidence at or above {threshold:.3f}")

        def _producer():
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if stop_requested_callback and stop_requested_callback():
                            stop_event.set()
                            break
                        if stop_event.is_set():
                            break

                        producer_state["input_count"] += 1
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        item_cleaned = validate_json(item)
                        if item_cleaned:
                            seq = producer_state["entry_count"]
                            producer_state["entry_count"] += 1
                            prepared = _prepare_conversation_entry(
                                seq,
                                item_cleaned,
                                split_token_limit,
                                status_update_callback,
                            )
                            prep_queue.put(prepared)

                        if status_update_callback and producer_state["input_count"] % 100 == 0:
                            percent_done = (producer_state["input_count"] / total_lines) * 100
                            remaining = total_lines - producer_state["input_count"]
                            status_update_callback(
                                f"Read {producer_state['input_count']}/{total_lines} lines ({percent_done:.1f}%). Remaining to read: {remaining}"
                            )
            except Exception as e:
                producer_state["error"] = e
            finally:
                try:
                    prep_queue.put(sentinel, timeout=1)
                except Exception:
                    pass

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        active_entries = {}
        chunk_queue = []
        finalized_entries = {}
        next_emit_seq = 0
        producer_done = False
        zero_score_entries = 0
        last_status_finalized = 0

        def _finish_entry(seq, is_refusal):
            nonlocal zero_score_entries
            global total_kept_count, total_refusal_count, total_clean_count
            entry = active_entries.pop(seq, None)
            if entry is None:
                return
            if entry["score_count"] == 0:
                zero_score_entries += 1
            if is_refusal:
                total_refusal_count += 1
                finalized_entries[seq] = None
            else:
                total_clean_count += 1
                total_kept_count += 1
                finalized_entries[seq] = entry["conversation"]

        def _flush_writer():
            nonlocal next_emit_seq
            while next_emit_seq in finalized_entries:
                conversation = finalized_entries.pop(next_emit_seq)
                if conversation is not None:
                    json_str = json.dumps(conversation, ensure_ascii=False)
                    json_str = clean_text(json_str)
                    if validate_utf8(json_str):
                        writer.write(json_str + "\n")
                next_emit_seq += 1

        while not producer_done or active_entries or chunk_queue:
            if stop_requested_callback and stop_requested_callback():
                stop_event.set()
                raise ProcessingCancelled()

            drained_item = False
            while not producer_done and len(chunk_queue) < microbatch_size * 4:
                try:
                    prepared_entry = prep_queue.get(timeout=0.05)
                except queue.Empty:
                    break
                drained_item = True
                if prepared_entry is sentinel:
                    producer_done = True
                    break

                seq = prepared_entry["seq"]
                chunks = prepared_entry["chunks"]
                active_entries[seq] = {
                    "conversation": prepared_entry["conversation"],
                    "remaining": len(chunks),
                    "score_count": 0,
                    "refused": False,
                }
                if chunks:
                    chunk_queue.extend((seq, chunk) for chunk in chunks)
                else:
                    _finish_entry(seq, False)

            if chunk_queue and (len(chunk_queue) >= microbatch_size or producer_done or not drained_item):
                batch = chunk_queue[:microbatch_size]
                del chunk_queue[:microbatch_size]
                scores = predict(
                    [chunk for _, chunk in batch],
                    microbatch_size,
                    status_update_callback,
                    stop_requested_callback,
                )

                for (seq, _chunk), score in zip(batch, scores):
                    entry = active_entries.get(seq)
                    if entry is None:
                        continue
                    entry["score_count"] += 1
                    entry["remaining"] -= 1
                    if score >= threshold:
                        entry["refused"] = True
                        skipped = sum(1 for queued_seq, _ in chunk_queue if queued_seq == seq)
                        if skipped:
                            chunk_queue = [(queued_seq, chunk) for queued_seq, chunk in chunk_queue if queued_seq != seq]
                            entry["remaining"] -= skipped
                    if entry["remaining"] <= 0:
                        _finish_entry(seq, entry["refused"])

                _flush_writer()

                finalized_count = total_refusal_count + total_clean_count
                if counts_update_callback:
                    counts_update_callback(total_refusal_count, total_clean_count)
                if progress_update_callback and total_lines > 0:
                    percent_done = int((finalized_count / total_lines) * 100)
                    progress_update_callback(max(0, min(99, percent_done)))
                if status_update_callback and total_lines > 0 and finalized_count - last_status_finalized >= 100:
                    last_status_finalized = finalized_count
                    write_ratio = (total_clean_count / total_lines) * 100
                    status_update_callback(
                        f"Written {total_clean_count}/{total_lines} entries to output ({write_ratio:.1f}%)."
                    )
            elif producer_done and not chunk_queue:
                _flush_writer()
                break

        stop_event.set()
        producer_thread.join(timeout=2)

        if producer_state["error"] is not None:
            raise producer_state["error"]

        total_input_count = producer_state["input_count"]

        writer.close()

        if counts_update_callback:
            counts_update_callback(total_refusal_count, total_clean_count)
        if progress_update_callback:
            progress_update_callback(100)
        if status_update_callback:
            if zero_score_entries > 0:
                status_update_callback(f"Warning: {zero_score_entries} entries had no scorable assistant text.")
            status_update_callback(
                f"Filtering complete. Compliance kept: {total_clean_count} | Refusals removed: {total_refusal_count} | Total: {total_input_count}. Output: {output_file}"
            )

    except ProcessingCancelled:
        try:
            stop_event.set()
        except Exception:
            pass
        if status_update_callback:
            status_update_callback("Streaming cancelled by user.")
    except Exception as e:
        try:
            stop_event.set()
        except Exception:
            pass
        if status_update_callback:
            status_update_callback(f"Streaming error: {e}")


def _split_text_for_model(text, tokenizer, token_limit):
    text = clean_text(text)
    if not text.strip():
        return []

    if tokenizer is None:
        return [text]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= token_limit:
        return [text]

    chunks = []
    for i in range(0, len(token_ids), token_limit):
        chunk_ids = token_ids[i:i + token_limit]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks if chunks else [text]


def _ensure_nlp(status_update_callback=None):
    global nlp
    if nlp is None:
        if status_update_callback:
            status_update_callback("Loading sentence splitter...")
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")
    return nlp


def _get_producer_tokenizer(status_update_callback=None):
    model_name = get_model_name()
    if model_name not in producer_tokenizers:
        if status_update_callback:
            status_update_callback("Loading CPU tokenizer for streaming preparation...")
        # Separate tokenizer instance for producer thread token counting/splitting.
        # Prevents Rust tokenizer borrow conflicts with concurrent GPU inference tokenization.
        producer_tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
    return producer_tokenizers[model_name]


def _is_assistant_role(role):
    role = str(role).strip().lower()
    return role in {"gpt", "assistant", "model"}


def _prepare_conversation_entry(seq, conversation, split_token_limit, status_update_callback=None):
    tokenizer = _get_producer_tokenizer(status_update_callback)
    split_token_limit = max(32, int(split_token_limit))
    assistant_turns = [
        clean_text(turn.get("value", ""))
        for turn in conversation.get("conversations", [])
        if _is_assistant_role(turn.get("from")) and turn.get("value") is not None
    ]
    assistant_turns = [text for text in assistant_turns if text]
    entry_text = " ".join(assistant_turns).strip()

    chunks = []
    if entry_text:
        if tokenizer is None:
            token_count = len(entry_text.split())
        else:
            token_count = len(tokenizer.encode(entry_text, add_special_tokens=False))

        if token_count <= split_token_limit:
            chunks.append(entry_text)
        else:
            nlp_local = _ensure_nlp(status_update_callback)
            docs = nlp_local.pipe(assistant_turns, batch_size=32)
            for doc in docs:
                for sent in doc.sents:
                    sent_text = clean_text(sent.text.strip())
                    if sent_text:
                        chunks.extend(_split_text_for_model(sent_text, tokenizer, split_token_limit))

    return {
        "seq": seq,
        "conversation": conversation,
        "chunks": chunks,
    }


def update_status(message, status_update_callback=None):
    if status_update_callback:
        status_update_callback(message)


def update_counts(refusal_count, clean_count, counts_update_callback=None):
    if counts_update_callback:
        counts_update_callback(refusal_count, clean_count)


def predict(inputs, microbatch_size=4, status_update_callback=None, stop_requested_callback=None):
    results = []
    model_name = get_model_name()

    refusal_idx = get_refusal_index()
    device = models[model_name].device

    use_autocast = device.type == "cuda" and inference_precision in {"fp16", "bf16"}
    if inference_precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    i = 0
    current_microbatch = max(1, int(microbatch_size))
    warned_oom = False
    while i < len(inputs):
        if stop_requested_callback and stop_requested_callback():
            raise ProcessingCancelled()
        chunk = inputs[i:i + current_microbatch]
        try:
            encoded_inputs = tokenizers[model_name](
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=get_model_max_tokens(),
            ).to(models[model_name].device)

            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        logits = models[model_name](**encoded_inputs).logits
                else:
                    logits = models[model_name](**encoded_inputs).logits

            if logits.ndim == 2 and logits.shape[1] >= 2:
                probs = torch.softmax(logits, dim=-1)
                predicted_ids = torch.argmax(probs, dim=-1)
                predicted_scores = torch.max(probs, dim=-1).values
                refusal_scores = torch.where(
                    predicted_ids == refusal_idx,
                    predicted_scores,
                    torch.zeros_like(predicted_scores),
                ).detach().cpu().tolist()
            else:
                probs = torch.sigmoid(logits)
                refusal_scores = probs.squeeze(-1).detach().cpu().tolist()
                if not isinstance(refusal_scores, list):
                    refusal_scores = [float(refusal_scores)]

            results.extend([float(score) for score in refusal_scores])
            i += len(chunk)
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" not in err or current_microbatch <= 1:
                raise

            new_microbatch = max(1, current_microbatch // 2)
            if new_microbatch == current_microbatch:
                raise
            current_microbatch = new_microbatch
            if models[model_name].device.type == "cuda":
                torch.cuda.empty_cache()
            if status_update_callback and not warned_oom:
                status_update_callback(
                    f"GPU memory pressure detected, reducing inference microbatch to {current_microbatch}"
                )
                warned_oom = True

    return results


def validate_json(conversation):
    if not isinstance(conversation, dict):
        return None
    valid_keys = {'conversations'}
    cleaned_conversation = {k: v for k, v in conversation.items() if k in valid_keys}
    if 'conversations' in cleaned_conversation:
        cleaned_conversation['conversations'] = [
            turn for turn in cleaned_conversation['conversations']
            if isinstance(turn, dict) and 'from' in turn and 'value' in turn
        ]
    return cleaned_conversation if cleaned_conversation else None


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    # Keep text intact; only strip NUL which can break tokenization/serialization.
    return text.replace('\x00', '')


def validate_utf8(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False
