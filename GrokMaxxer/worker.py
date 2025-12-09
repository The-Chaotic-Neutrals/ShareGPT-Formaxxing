import json
import os
import random
from datetime import datetime
import time
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None  # Optional dependency

# Ensure we can import as a package - add parent directory to path if needed
import sys
from pathlib import Path
_grokmaxxer_dir = Path(__file__).parent
_parent_dir = _grokmaxxer_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from GrokMaxxer.llm_helpers import (
    improve_entry,
    generate_new_entry,
    extend_entry,
    improve_and_extend_entry,
    generate_human_turns,
    rewrite_human_cache,
    generate_names,
)
from GrokMaxxer.schema import ensure_system_message, fix_sharegpt_entry, validate_sharegpt_entry

from xai_sdk import Client
import concurrent.futures
import threading
from datasets import load_dataset


def process_entry(entry, client, model, system_prompt, do_rewrite=False, extra_pairs=0, reply_in_character=False, dynamic_names_mode=False, names_list=None):
    """Pure processing function - no GUI logging or terminal colors"""
    result_entry = entry
    temp = random.uniform(0.69, 1.42)
    char_system_prompt = system_prompt
    
    if dynamic_names_mode and names_list:
        name_info = random.choice(names_list)
        name = name_info['name']
        category = name_info['category']
        char_system_prompt += f"\n\nRoleplay guidance: You are conversing with {name}, a {category}. Address {name} by name naturally throughout the conversation and personalize your responses accordingly."
    
    if do_rewrite and extra_pairs > 0:
        result_entry = improve_and_extend_entry(
            result_entry, client, model, char_system_prompt, extra_pairs, temperature=temp
        )
    elif do_rewrite:
        result_entry = improve_entry(result_entry, client, model, char_system_prompt, temperature=temp)
    elif extra_pairs > 0:
        result_entry = extend_entry(result_entry, client, model, char_system_prompt, extra_pairs, temperature=temp)
    
    if reply_in_character and char_system_prompt:
        result_entry = ensure_system_message(result_entry, char_system_prompt)

    fixed_entry = fix_sharegpt_entry(result_entry)
    if fixed_entry:
        result_entry = fixed_entry

    ok, _ = validate_sharegpt_entry(result_entry)
    return result_entry if ok else None


def worker(
    input_file,
    output_file,
    num_new,
    api_key,
    model_name,
    system_prompt,
    start_line,
    end_line,
    do_rewrite,
    extra_pairs,
    reply_in_character,
    dynamic_names_mode=False,
    rewrite_cache=False,
    q=None,
):

    raw_system_prompt = system_prompt

    if q:
        q.put(("log", "Worker thread started"))

    try:
        if not api_key:
            if q:
                q.put(("error", "No API key provided"))
            return

        client = None
        try:
            client = Client(api_key=api_key, timeout=300)
        except Exception as client_e:
            if q:
                q.put(("error", f"Failed to initialize xAI client: {client_e}"))
            return

        if model_name:
            models_list = [m.strip() for m in model_name.split(',') if m.strip()]
        else:
            models_list = ["grok-beta"]

        def pick_model():
            return random.choice(models_list)

        names_list = None
        if dynamic_names_mode:
            num_names_needed = max(10, num_new // 10)
            names_cache_file = "global_names_cache.json"
            names_list = []
            if os.path.exists(names_cache_file):
                try:
                    with open(names_cache_file, 'r', encoding='utf-8') as f:
                        names_list = json.load(f)
                    if q:
                        q.put(("log", f"Loaded {len(names_list)} cached names from {names_cache_file}"))
                except Exception as load_e:
                    if q:
                        q.put(("log", f"Failed to load names cache {names_cache_file}: {load_e}"))
            
            if len(names_list) < num_names_needed:
                to_gen = num_names_needed - len(names_list)
                if q:
                    q.put(("log", f"Names cache {len(names_list)} < target {num_names_needed}, generating {to_gen} more..."))
                new_names = generate_names(client, pick_model(), num_names=to_gen)
                names_list.extend(new_names)
                
                try:
                    with open(names_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(names_list, f, indent=2, ensure_ascii=False)
                    if q:
                        q.put(("log", f"Updated names cache {names_cache_file} with {len(names_list)} names"))
                except Exception as save_e:
                    if q:
                        q.put(("log", f"Failed to save names cache: {save_e}"))
            else:
                if q:
                    q.put(("log", f"Using existing names cache of {len(names_list)} names"))
            
            if q:
                q.put(("log", f"Dynamic names mode enabled: final pool {len(names_list)} names"))

        last_temp = 0.0
        last_temp_lock = threading.Lock()
        last_domain = None
        last_domain_lock = threading.Lock()

        if q:
            q.put(("log", "Starting simplified processing..."))

        file_size = 0
        if input_file and os.path.exists(input_file):
            try:
                file_size = os.path.getsize(input_file)
            except OSError:
                file_size = 0

        total_steps = file_size + max(num_new, 0)
        if total_steps <= 0:
            total_steps = 1

        progress_state = {"bytes": 0.0, "generated": 0.0}

        def update_progress():
            completed = progress_state["bytes"] + progress_state["generated"]
            progress = completed / total_steps if total_steps > 0 else 1.0
            if q:
                q.put(("progress", progress))

        skipped_unmodified_count = 0
        invalid_count = 0

        def log_entry_preview(label, obj):

            try:
                text = json.dumps(obj, ensure_ascii=False)
            except Exception:
                text = str(obj)
            max_len = 400
            if len(text) > max_len:
                text = text[:max_len] + "... (truncated)"
            if q:
                q.put(("log", f"{label}: {text}"))

        if start_line is None or start_line < 1:
            start_line = 1
        if end_line is not None and end_line < start_line:
            raise ValueError("End line cannot be less than start line")

        example_entries = []
        if input_file and os.path.exists(input_file):
            try:
                dataset = load_dataset("json", data_files=input_file, split="train", streaming=True)
                count = 0
                for ex in dataset:
                    ok, _ = validate_sharegpt_entry(ex)
                    if ok:
                        example_entries.append(ex)
                        count += 1
                        if count >= 5:
                            break
            except Exception as e:
                if q:
                    q.put(("log", f"Error loading examples with datasets: {e}. Falling back to manual."))
                with open(input_file, "r", encoding="utf-8") as ex_f:
                    for _ in range(5):
                        line = ex_f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ex = json.loads(line)
                            ok, _ = validate_sharegpt_entry(ex)
                            if ok:
                                example_entries.append(ex)
                        except json.JSONDecodeError:
                            continue

        needs_processing = do_rewrite or extra_pairs > 0

        should_process_input = bool(input_file and os.path.exists(input_file) and (do_rewrite or extra_pairs > 0 or num_new == 0))
        if should_process_input:
            if q:
                q.put(("log", "Processing existing entries sequentially..."))
            with open(output_file, "w", encoding="utf-8") as outfile:
                try:
                    dataset = load_dataset("json", data_files=input_file, split="train", streaming=True)
                    for line_num, ex in enumerate(dataset, start=1):
                        if end_line is not None and line_num > end_line:
                            break
                        progress_state["bytes"] += len(json.dumps(ex).encode("utf-8"))
                        update_progress()

                        ok, _ = validate_sharegpt_entry(ex)
                        if not ok:
                            if q:
                                q.put(("log", f"Skipping invalid entry on line {line_num}"))
                            continue

                        in_range = (line_num >= start_line)

                        if in_range and needs_processing:

                            result = process_entry(ex, client, pick_model(), raw_system_prompt, do_rewrite, extra_pairs, reply_in_character, dynamic_names_mode, names_list)
                            if result is None:
                                invalid_count += 1
                            else:
                                try:
                                    json.dump(result, outfile, ensure_ascii=False)
                                    outfile.write("\n")
                                    outfile.flush()
                                    log_entry_preview(f"[PROCESSED ENTRY line {line_num}]", result)
                                except Exception as write_e:
                                    if q:
                                        q.put(("log", f"Error writing processed entry line {line_num}: {write_e}"))
                                    invalid_count += 1
                        elif not in_range:

                            try:
                                json.dump(ex, outfile, ensure_ascii=False)
                                outfile.write("\n")
                                outfile.flush()
                            except Exception as write_e:
                                if q:
                                    q.put(("log", f"Error writing non-range entry line {line_num}: {write_e}"))
                        else:

                            skipped_unmodified_count += 1

                except Exception as e:
                    if q:
                        q.put(("log", f"Error loading dataset: {e}. Falling back to manual reading."))

                    with open(input_file, "r", encoding="utf-8") as infile:
                        for line_num, line in enumerate(infile, start=1):
                            if end_line is not None and line_num > end_line:
                                break
                            try:
                                progress_state["bytes"] = infile.tell()
                            except (OSError, IOError):
                                progress_state["bytes"] += len(line.encode("utf-8", errors="ignore"))

                            raw_line = line
                            stripped = line.strip()
                            if not stripped:

                                try:
                                    outfile.write(raw_line)
                                    outfile.flush()
                                except Exception as write_e:
                                    if q:
                                        q.put(("log", f"Error writing empty line {line_num}: {write_e}"))
                                update_progress()
                                continue

                            entry = None
                            try:
                                entry = json.loads(stripped)
                            except json.JSONDecodeError as e:
                                try:
                                    if repair_json is None:
                                        raise ImportError("json_repair module not available")
                                    repaired = repair_json(raw_line)
                                    repaired_stripped = repaired.strip()
                                    if repaired_stripped:
                                        entry = json.loads(repaired_stripped)
                                        if q:
                                            q.put(("log", f"Repaired malformed JSON on line {line_num}"))
                                except Exception:
                                    pass
                                if entry is None:
                                    if q:
                                        q.put(("log", f"Skipping malformed JSON on line {line_num}: {e}"))
                                    update_progress()
                                    continue

                            ok, _ = validate_sharegpt_entry(entry)
                            if not ok:
                                continue

                            in_range = (line_num >= start_line)

                            if in_range and needs_processing:

                                result = process_entry(entry, client, pick_model(), raw_system_prompt, do_rewrite, extra_pairs, reply_in_character, dynamic_names_mode, names_list)
                                if result is None:
                                    invalid_count += 1
                                else:
                                    try:
                                        json.dump(result, outfile, ensure_ascii=False)
                                        outfile.write("\n")
                                        outfile.flush()
                                        log_entry_preview(f"[PROCESSED ENTRY line {line_num}]", result)
                                    except Exception as write_e:
                                        if q:
                                            q.put(("log", f"Error writing processed entry line {line_num}: {write_e}"))
                                        invalid_count += 1
                            elif not in_range:
                                try:
                                    json.dump(entry, outfile, ensure_ascii=False)
                                    outfile.write("\n")
                                    outfile.flush()
                                except Exception as write_e:
                                    if q:
                                        q.put(("log", f"Error writing non-range entry line {line_num}: {write_e}"))
                            else:

                                skipped_unmodified_count += 1

                            update_progress()

                progress_state["bytes"] = file_size
                update_progress()
        else:
            if input_file:
                if q:
                    q.put(("log", "Input file provided for examples only; skipping copy to output (generation-only mode)."))
                progress_state["bytes"] = file_size
                update_progress()


        if num_new > 0:
            if q:
                q.put(("log", f"Generating {num_new} new entries in parallel with varying temperatures..."))
            
            cache_file = "global_human_cache.json"
            human_turns = []
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        human_turns = json.load(f)
                    if q:
                        q.put(("log", f"Loaded {len(human_turns)} cached human turns from {cache_file}"))
                    
                    if rewrite_cache:
                        if q:
                            q.put(("log", "Rewriting existing human cache entries..."))
                        human_turns = rewrite_human_cache(client, pick_model(), raw_system_prompt, cache_file, q=q)
                        if q:
                            q.put(("log", f"Rewrote cache: now {len(human_turns)} quality human turns"))
                    else:
                        if q:
                            q.put(("log", f"Using existing human cache without rewrite"))
                except Exception as load_e:
                    if q:
                        q.put(("log", f"Failed to load cache {cache_file}: {load_e}"))
            
            target_pool = max(100, num_new)
            if len(human_turns) < target_pool:
                to_gen = max(0, target_pool - len(human_turns))
                if to_gen > 0:
                    if q:
                        q.put(("log", f"Cache {len(human_turns)} < target {target_pool} for {num_new} entries, generating {to_gen} more..."))
                    new_turns = generate_human_turns(
                        client,
                        pick_model(),
                        raw_system_prompt,
                        num_turns=to_gen,
                        temperature=1.0
                    )
                    human_turns.extend(new_turns)
                

                # NO deduplication or cache overwrite after rewrite_human_cache - preserve ALL rewritten entries
                if q:
                    q.put(("log", f"Preserving {len(human_turns)} rewritten human turns from cache (no deduplication)"))
            else:
                if q:
                    q.put(("log", f"Using existing cache of {len(human_turns)} human turns (no new generation needed)"))
            
            random.shuffle(human_turns)
            
            non_empty_items = [item for item in human_turns if 'message' in item and item['message'].strip() and 'domain' in item]
            human_pool_lock = threading.Lock()
            
            if q:
                q.put(("log", f"Human turns pool ready: {len(human_turns)} total, {len(non_empty_items)} non-empty"))
            
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            mode = "a" if should_process_input else "w"
            if q:
                q.put(("log", f"Opening output file: {output_file} in mode '{mode}'"))
            
            def generate_single_entry(entry_idx):
                nonlocal last_temp, last_domain, dynamic_names_mode, names_list
                try:
                    temp = None
                    with last_temp_lock:
                        attempts = 0
                        while attempts < 100:
                            temp = random.uniform(0.69, 1.42)
                            if last_temp is None or abs(temp - last_temp) >= 0.05:
                                last_temp = temp
                                break
                            attempts += 1
                        if temp is None:
                            temp = random.uniform(0.69, 1.42)
                            last_temp = temp
                    start_time = time.perf_counter()
                    

                    with human_pool_lock:
                        starting_human = None
                        domain = None
                        max_tries = min(20, len(non_empty_items) + 1)
                        for _ in range(max_tries):
                            if not non_empty_items:
                                break
                            item_idx = random.randint(0, len(non_empty_items) - 1)
                            item = non_empty_items[item_idx]
                            candidate_domain = item['domain']
                            if candidate_domain != last_domain:
                                non_empty_items.pop(item_idx)
                                domain = candidate_domain
                                starting_human = item['message']
                                break
                        if starting_human is None and non_empty_items:
                            item_idx = random.randint(0, len(non_empty_items) - 1)
                            item = non_empty_items.pop(item_idx)
                            domain = item['domain']
                            starting_human = item['message']
                    with last_domain_lock:
                        last_domain = domain
                    
                    char_system_prompt = raw_system_prompt
                    if dynamic_names_mode and names_list:
                        name_info = random.choice(names_list)
                        name = name_info['name']
                        category = name_info['category']
                        char_system_prompt += f"\n\nRoleplay guidance: You are conversing with {name}, a {category}. Address {name} by name naturally throughout the conversation and personalize your responses accordingly."
                        if q:
                            q.put(("log", f"Applied dynamic roleplay for entry {entry_idx}: {name} ({category})"))
                    
                    new_entry = generate_new_entry(
                        client,
                        pick_model(),
                        char_system_prompt,
                        example_entries,
                        starting_human=starting_human,
                        temperature=temp,
                    )
                    api_time = time.perf_counter() - start_time
                    starting_info = f" (starting: '{starting_human[:50]}...', domain: {last_domain})" if starting_human else ""
                    if q:
                        q.put(("log", f"Generated entry {entry_idx} (temp={temp:.2f}, took {api_time:.2f}s{starting_info})"))
                    
                    if reply_in_character and char_system_prompt:
                        new_entry = ensure_system_message(new_entry, char_system_prompt)
                        if q:
                            q.put(("log", f"Applied dynamic system prompt with name to generated entry {entry_idx}"))

                    fixed_entry = fix_sharegpt_entry(new_entry)
                    if fixed_entry:
                        new_entry = fixed_entry
                        if q:
                            q.put(("log", f"Auto-fixed schema issues for generated entry {entry_idx}"))

                    ok, errors = validate_sharegpt_entry(new_entry)
                    if not ok:
                        if q:
                            q.put(("log", f"Validation failed for generated entry {entry_idx}: {errors}"))
                        return None, entry_idx

                    return new_entry, entry_idx
                except Exception as e:
                    if q:
                        q.put(("log", f"Error generating entry {entry_idx}: {e}"))
                    return None, entry_idx

            try:
                with open(output_file, mode, encoding="utf-8") as outfile:
                    written_count = 0
                    error_count = 0
                    
                    max_workers = min(5, num_new)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(generate_single_entry, i + 1): i + 1
                            for i in range(num_new)
                        }
                        
                        for future in concurrent.futures.as_completed(futures):
                            entry_idx = futures[future]
                            try:
                                new_entry, idx = future.result()
                                if new_entry is None:
                                    error_count += 1
                                    continue

                                json.dump(new_entry, outfile, ensure_ascii=False)
                                outfile.write("\n")
                                outfile.flush()
                                log_entry_preview(f"[GENERATED ENTRY {idx}]", new_entry)
                                written_count += 1
                                progress_state["generated"] += 1
                                update_progress()
                            except Exception as exc:
                                if q:
                                    q.put(("log", f"Future for entry {entry_idx} raised {exc}"))
                                error_count += 1

                    if q:
                        q.put(("log", f"Parallel generation complete: {written_count} written, {error_count} errors"))
                        q.put(("log", f"File mode was: {mode}, file exists: {os.path.exists(output_file)}"))
            except Exception as file_e:
                if q:
                    q.put(("log", f"Error opening/writing to output file {output_file}: {file_e}"))

        if q:
            q.put(("progress", 1.0))
            q.put(("success", "Processing completed successfully."))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if q:
            q.put(("error", str(e)))
            q.put(("log", f"Fatal error in worker: {e}"))
            q.put(("log", f"Full traceback:\n{tb}"))
