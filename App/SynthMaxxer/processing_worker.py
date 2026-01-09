"""
Processing worker for SynthMaxxer
Handles JSONL file processing, improvement, extension, and generation
"""
import json
import os
import random
import time
import threading
import concurrent.futures

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

import sys
from pathlib import Path
_synthmaxxer_dir = Path(__file__).parent
_parent_dir = _synthmaxxer_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from App.SynthMaxxer.llm_helpers import (
    improve_entry,
    generate_new_entry,
    extend_entry,
    improve_and_extend_entry,
    generate_names,
)
from App.SynthMaxxer.llm_client import create_client
from App.SynthMaxxer.schema import (
    from_sharegpt,
    to_sharegpt,
    fix_messages,
    add_system_message,
    validate_messages,
)

SYNTHMAXXER_DIR = Path(__file__).parent
GLOBAL_NAMES_CACHE_FILE = str(SYNTHMAXXER_DIR / "global_names_cache.json")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def process_entry(entry, client, model, system_prompt, do_rewrite=False, extra_pairs=0, reply_in_character=False, dynamic_names_mode=False, names_list=None):
    """Process a single entry - rewrite and/or extend"""
    temp = random.uniform(0.69, 1.42)
    char_system_prompt = system_prompt
    
    if dynamic_names_mode and names_list:
        name_info = random.choice(names_list)
        name = name_info['name']
        age = name_info.get('age', 25)
        char_system_prompt += f"\n\nRoleplay: You are conversing with {name} (age {age}). Address them by name naturally."
    
    try:
        if do_rewrite and extra_pairs > 0:
            result = improve_and_extend_entry(entry, client, model, char_system_prompt, extra_pairs, temperature=temp)
        elif do_rewrite:
            result = improve_entry(entry, client, model, char_system_prompt, temperature=temp)
        elif extra_pairs > 0:
            result = extend_entry(entry, client, model, char_system_prompt, extra_pairs, temperature=temp)
        else:
            result = entry
    except Exception:
        return None
    
    # Add system message if requested
    if reply_in_character and char_system_prompt:
        messages = from_sharegpt(result)
        if messages:
            messages = add_system_message(messages, char_system_prompt)
            result = to_sharegpt(messages)
    
    # Validate result
    messages = from_sharegpt(result)
    if not messages:
        return None
    ok, _ = validate_messages(messages)
    return result if ok else None


def processing_worker(
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
    q=None,
    stop_flag=None,
    api_type="OpenAI Chat Completions",
    endpoint=None,
):
    """Main processing worker"""
    def log(msg):
        if q:
            q.put(("log", msg))
        else:
            print(msg)
    
    log(f"Processing worker started (API: {api_type})")
    
    try:
        if not api_key:
            if q:
                q.put(("error", "No API key provided"))
            return
        
        # Create client
        client = create_client(api_key=api_key, api_type=api_type, endpoint=endpoint, timeout=300)
        log(f"Created LLM client for {api_type}")
        
        # Model selection
        models = [m.strip() for m in model_name.split(',') if m.strip()] if model_name else ["gpt-4o-mini"]
        pick_model = lambda: random.choice(models)
        
        # Dynamic names
        names_list = None
        if dynamic_names_mode:
            names_list = []
            if os.path.exists(GLOBAL_NAMES_CACHE_FILE):
                try:
                    with open(GLOBAL_NAMES_CACHE_FILE, 'r', encoding='utf-8') as f:
                        names_list = json.load(f)
                    log(f"Loaded {len(names_list)} cached names")
                except:
                    pass
            
            needed = max(10, num_new // 10) - len(names_list)
            if needed > 0:
                log(f"Generating {needed} names...")
                new_names = generate_names(client, pick_model(), num_names=needed)
                names_list.extend(new_names)
                try:
                    with open(GLOBAL_NAMES_CACHE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(names_list, f, indent=2, ensure_ascii=False)
                except:
                    pass
        
        # Line range
        if start_line is None or start_line < 1:
            start_line = 1
        if end_line is not None and end_line < start_line:
            raise ValueError("End line cannot be less than start line")
        
        # Load examples for generation
        example_entries = []
        if input_file and os.path.exists(input_file):
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    for _ in range(5):
                        line = f.readline()
                        if not line:
                            break
                        try:
                            entry = json.loads(line.strip())
                            messages = from_sharegpt(entry)
                            if messages:
                                ok, _ = validate_messages(messages)
                                if ok:
                                    example_entries.append(entry)
                        except:
                            pass
            except:
                pass
        
        # Validation
        if not input_file and num_new == 0:
            if q:
                q.put(("error", "No input file and no new entries to generate"))
            return
        
        if input_file and not os.path.exists(input_file):
            if q:
                q.put(("error", f"Input file not found: {input_file}"))
            return
        
        # Ensure output dir
        if output_file:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        
        needs_processing = do_rewrite or extra_pairs > 0
        should_process = bool(input_file and os.path.exists(input_file) and (needs_processing or num_new == 0))
        
        # Process existing file
        if should_process:
            log("Processing existing entries...")
            with open(output_file, "w", encoding="utf-8") as outfile:
                with open(input_file, "r", encoding="utf-8") as infile:
                    for line_num, line in enumerate(infile, start=1):
                        if stop_flag and stop_flag.is_set():
                            if q:
                                q.put(("stopped", "Stopped by user"))
                            break
                        if end_line and line_num > end_line:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            entry = json.loads(line)
                        except:
                            if repair_json:
                                try:
                                    entry = json.loads(repair_json(line))
                                except:
                                    log(f"Skipping malformed line {line_num}")
                                    continue
                            else:
                                continue
                        
                        # Validate entry
                        messages = from_sharegpt(entry)
                        if not messages:
                            continue
                        
                        in_range = line_num >= start_line
                        
                        if in_range and needs_processing:
                            result = process_entry(entry, client, pick_model(), system_prompt, do_rewrite, extra_pairs, reply_in_character, dynamic_names_mode, names_list)
                            if result:
                                json.dump(result, outfile, ensure_ascii=False)
                                outfile.write("\n")
                                log(f"[PROCESSED {line_num}]")
                        else:
                            json.dump(entry, outfile, ensure_ascii=False)
                            outfile.write("\n")
        
        # Generate new entries
        if num_new > 0:
            log(f"Generating {num_new} new entries...")
            
            mode = "a" if should_process else "w"
            
            def generate_one(idx):
                temp = random.uniform(0.69, 1.42)
                char_prompt = system_prompt
                
                if dynamic_names_mode and names_list:
                    name_info = random.choice(names_list)
                    char_prompt += f"\n\nRoleplay: Conversing with {name_info['name']} (age {name_info.get('age', 25)})."
                
                try:
                    entry = generate_new_entry(client, pick_model(), char_prompt, example_entries, temperature=temp)
                    
                    if reply_in_character and char_prompt:
                        messages = from_sharegpt(entry)
                        if messages:
                            messages = add_system_message(messages, char_prompt)
                            entry = to_sharegpt(messages)
                    
                    # Validate
                    messages = from_sharegpt(entry)
                    if messages:
                        ok, _ = validate_messages(messages)
                        if ok:
                            return entry, idx
                except Exception as e:
                    log(f"Error generating {idx}: {e}")
                return None, idx
            
            with open(output_file, mode, encoding="utf-8") as outfile:
                written = 0
                errors = 0
                
                workers = min(5, num_new)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(generate_one, i + 1): i + 1 for i in range(num_new)}
                    
                    for future in concurrent.futures.as_completed(futures):
                        if stop_flag and stop_flag.is_set():
                            if q:
                                q.put(("stopped", "Stopped by user"))
                            break
                        
                        try:
                            entry, idx = future.result()
                            if entry:
                                json.dump(entry, outfile, ensure_ascii=False)
                                outfile.write("\n")
                                outfile.flush()
                                log(f"[GENERATED {idx}]")
                                written += 1
                            else:
                                errors += 1
                        except Exception as e:
                            log(f"Future error: {e}")
                            errors += 1
                
                log(f"Generation complete: {written} written, {errors} errors")
        
        if q:
            q.put(("progress", 1.0))
            q.put(("success", "Processing completed"))
    
    except Exception as e:
        import traceback
        if q:
            q.put(("error", str(e)))
            q.put(("log", f"Traceback:\n{traceback.format_exc()}"))
