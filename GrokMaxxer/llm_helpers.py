import json
from xai_sdk import Client
from xai_sdk.chat import system
import os
import time
from json_repair import repair_json

# Ensure we can import as a package - add parent directory to path if needed
import sys
from pathlib import Path
_grokmaxxer_dir = Path(__file__).parent
_parent_dir = _grokmaxxer_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from GrokMaxxer.model_config import SCHEMA_DESCRIPTION
from GrokMaxxer.schema import extract_json_object, validate_sharegpt_entry, ensure_system_message, extract_json_array, validate_conversations, fix_sharegpt_entry

import re


def _with_system_prefix(system_prompt: str) -> str:
    if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
        return system_prompt.strip() + "\n\n"
    return ""


def _normalize_domain(raw, valid_domains):
    if not isinstance(raw, str):
        return None
    raw_stripped = raw.strip()

    if raw_stripped in valid_domains:
        return raw_stripped

    norm = raw_stripped.replace("&", "and").replace("_", " ").lower()

    for d in valid_domains:
        d_norm = d.replace("&", "and").replace("_", " ").lower()
        if norm == d_norm:
            return d

    return "Other"


def improve_entry(
    entry,
    client,
    model,
    system_prompt,
    temperature: float | None = None,
):

    schema_instructions = (
        "You are improving chat-style fine-tuning data in a ShareGPT-like format.\n"
        "The conversations is a list of messages.\n"
        "Each message has:\n"
        "  - \"from\": one of [\"system\", \"human\", \"gpt\"]\n"
        "  - \"value\": string content\n\n"
        "CRITICAL SCHEMA RULES - MUST FOLLOW EXACTLY:\n"
        "  1) Do NOT add, remove, or reorder messages.\n"
        "  2) Do NOT change any \"from\" fields.\n"
        "  3) Optional leading \"system\" message, then STRICT alternation: human, gpt, human, gpt, ...\n"
        "  4) Final message MUST be from \"gpt\" (not human).\n"
        "  5) Only rewrite \"value\" strings for clarity, coherence, style, and diversity (vary human phrasing and topics to avoid repetition).\n"
        "  6) Output MUST be valid JSON array of the improved messages.\n"
        "  7) No markdown, no explanations, ONLY the JSON array.\n\n"
        "Violations will cause rejection. Double-check structure before outputting.\n"
    )

    prefix = _with_system_prefix(system_prompt)
    base_prompt = (
        prefix
        + schema_instructions
        + "\nOriginal conversations:\n"
        + json.dumps(entry["conversations"], ensure_ascii=False)
    )

    chat = client.chat.create(model=model, temperature=temperature)
    chat.append(system(base_prompt))
    response = chat.sample()
    content = response.content
    repaired = repair_json(content) or content
    try:
        parsed = json.loads(repaired)
    except Exception as e:
        try:
            improved_conv = extract_json_array(repaired)
        except ValueError:
            raise ValueError(f"Could not parse JSON from model response: {e}")
    else:
        if isinstance(parsed, list):
            improved_conv = parsed
        elif isinstance(parsed, dict) and "conversations" in parsed:
            improved_conv = parsed["conversations"]
        else:
            try:
                improved_conv = extract_json_array(repaired)
            except ValueError:
                raise ValueError("Model JSON did not contain a conversations array")

    if len(improved_conv) != len(entry["conversations"]):
        raise ValueError(f"Improved conversations length mismatch: expected {len(entry['conversations'])}, got {len(improved_conv)}")

    # Try to fix the improved conversations
    fixed_entry = fix_sharegpt_entry({"conversations": improved_conv})
    if fixed_entry is None:
        # Try a more aggressive fix: ensure all messages have required fields
        aggressive_fix_conv = []
        for i, msg in enumerate(improved_conv):
            if not isinstance(msg, dict):
                continue
            # Ensure 'from' and 'value' exist
            if "from" not in msg:
                # Try to infer from position
                if i == 0 and entry["conversations"][0].get("from") == "system":
                    msg["from"] = "system"
                elif (i - (1 if entry["conversations"][0].get("from") == "system" else 0)) % 2 == 0:
                    msg["from"] = "human"
                else:
                    msg["from"] = "gpt"
            if "value" not in msg or not isinstance(msg.get("value"), str):
                msg["value"] = msg.get("value", " ") if isinstance(msg.get("value"), str) else " "
            aggressive_fix_conv.append(msg)
        
        if len(aggressive_fix_conv) >= 2:
            fixed_entry = fix_sharegpt_entry({"conversations": aggressive_fix_conv})
        
        if fixed_entry is None:
            # Last resort: try to preserve original structure but with improved values where possible
            fallback_conv = []
            original_conv = entry["conversations"]
            for i, orig_msg in enumerate(original_conv):
                if i < len(improved_conv) and isinstance(improved_conv[i], dict):
                    # Use improved value if available, but preserve original structure
                    fallback_msg = {
                        "from": orig_msg.get("from", "human" if i % 2 == 0 else "gpt"),
                        "value": improved_conv[i].get("value", orig_msg.get("value", " "))
                    }
                else:
                    fallback_msg = orig_msg.copy()
                fallback_conv.append(fallback_msg)
            
            fixed_entry = fix_sharegpt_entry({"conversations": fallback_conv})
            if fixed_entry is None:
                raise ValueError(
                    f"Could not auto-fix improved conversations. "
                    f"Original length: {len(entry['conversations'])}, "
                    f"Improved length: {len(improved_conv)}, "
                    f"After aggressive fix length: {len(aggressive_fix_conv) if 'aggressive_fix_conv' in locals() else 0}"
                )
    
    improved_conv = fixed_entry["conversations"]

    ok, err = validate_conversations(improved_conv)
    if not ok:
        raise ValueError(f"Invalid improved conversations after auto-fix: {err}")

    return {"conversations": improved_conv}


def generate_new_entry(
    client,
    model,
    system_prompt,
    example_entries=None,
    starting_human: str | None = None,
    temperature: float | None = None,
):

    if example_entries is None:
        example_entries = []
    example_entries = example_entries[:5]

    simple_examples = [ex["conversations"] for ex in example_entries]

    examples_str = ""
    if simple_examples:
        examples_str = "Here are some example conversations:\n" + "\n\n---\n\n".join(
            json.dumps(s_ex, ensure_ascii=False, indent=2) for s_ex in simple_examples
        ) + "\n\n"

    schema_instructions = (
        "You are generating new dataset entries in a ShareGPT-like format for fine-tuning.\n"
        "Output a JSON array of messages for the conversation.\n"
        "Each message has:\n"
        "  - \"from\": one of [\"system\", \"human\", \"gpt\"]\n"
        "  - \"value\": string content\n\n"
        "CRITICAL SCHEMA RULES - MUST FOLLOW EXACTLY:\n"
        "  1) Optional single \"system\" message at start.\n"
        "  2) Then STRICT alternation: human, gpt, human, gpt, ...\n"
        "  3) At least 2 messages (human + gpt minimum); end with \"gpt\" (not human).\n"
        "  4) Produce coherent, self-contained dialogue fitting the system prompt and examples, with diverse human messages (vary phrasing, topics, styles to avoid repetition).\n"
        "  5) Output MUST be valid JSON array ONLY. No other fields or wrapper.\n"
        "  6) No markdown, no explanations, no code fences, ONLY the raw JSON array. Ensure it is parseable JSON.\n"
        "  7) Double-check: the JSON must start with [ and end with ].\n\n"
    )

    prefix = _with_system_prefix(system_prompt)
    base_prompt = (
        prefix
        + schema_instructions
        + examples_str
    )

    if starting_human:
        base_prompt += (
            f"Start the conversation with this specific human message: \"{starting_human}\"\n"
            "Then generate the corresponding gpt response, and continue the alternation as needed (at least one gpt response).\n"
            "Ensure the full conversation fits the system prompt context.\n"
        )
    else:
        base_prompt += "Now generate ONE new conversation:\n"

    chat = client.chat.create(model=model, temperature=temperature)
    chat.append(system(base_prompt))
    response = chat.sample()
    content = response.content
    
    repaired = repair_json(content)
    if repaired:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, list):
                new_conv = parsed
            else:
                new_conv = parsed.get("conversations", [])
        except:
            new_conv = []
    else:
        try:
            new_conv = extract_json_array(content)
        except ValueError:
            raise ValueError("Could not parse or repair JSON array from response")

    fixed_entry = fix_sharegpt_entry({"conversations": new_conv})
    if fixed_entry is None:
        raise ValueError("Could not auto-fix generated conversations")
    new_conv = fixed_entry["conversations"]

    if new_conv and new_conv[-1]["from"] == "human":
        gpt_value = generate_single_gpt_response(client, model, system_prompt, new_conv, temperature)
        new_conv.append({"from": "gpt", "value": gpt_value})

    ok, err = validate_conversations(new_conv)
    if not ok:
        raise ValueError(f"Post-fix conversations still invalid: {err}")

    return {"conversations": new_conv}


def extend_entry(
    entry,
    client,
    model,
    system_prompt,
    num_pairs,
    example_entries=None,
    temperature: float | None = None,
):

    if num_pairs <= 0:
        return entry

    ok, err = validate_sharegpt_entry(entry)
    if not ok:
        raise ValueError(f"Cannot extend invalid ShareGPT entry: {err}")
    conv = entry.get("conversations", [])
    if not conv or conv[-1].get("from") != "gpt":
        last_role = conv[-1].get("from") if conv else "none"
        raise ValueError(f"Cannot extend conversation ending in '{last_role}', expected 'gpt'")

    total_new = num_pairs * 2

    prefix = _with_system_prefix(system_prompt)

    if example_entries is None:
        example_entries = []
    example_entries = example_entries[:25]

    simple_examples = [ex["conversations"] for ex in example_entries]

    examples_str = ""
    if simple_examples:
        examples_str = "Here are some example conversations for style reference:\n" + "\n\n---\n\n".join(
            json.dumps(s_ex, ensure_ascii=False, indent=2) for s_ex in simple_examples
        ) + "\n\n"

    schema_instructions = (
        "You are extending an existing chat-style conversation in a ShareGPT-like format.\n"
        "The existing conversations is a list of messages.\n"
        "Each message has:\n"
        "  - \"from\": one of [\"system\", \"human\", \"gpt\"]\n"
        "  - \"value\": string content\n\n"
        "CRITICAL SCHEMA RULES - MUST FOLLOW EXACTLY:\n"
        "  1) Do NOT repeat, modify, or reorder existing messages.\n"
        "  2) ONLY generate NEW messages that logically continue the conversation.\n"
        "  3) Existing ends with 'gpt'; first new MUST be 'human', then alternate human/gpt.\n"
        f"  4) Generate EXACTLY {total_new} new messages, ending with 'gpt'.\n"
        "  5) Content must be coherent, consistent with existing dialogue, and diverse (vary human phrasing, introduce new topics or angles to avoid repetition).\n"
        "  6) Output ONLY a JSON array of the NEW messages (not existing).\n"
        "  7) Valid JSON array, no markdown, no extra text.\n\n"
        "Violations will cause rejection. Ensure exact count and role sequence.\n"
    )

    prompt = (
        prefix
        + schema_instructions
        + examples_str
        + "Existing conversations:\n"
        + json.dumps(entry["conversations"], ensure_ascii=False, indent=2)
        + f"\n\nGenerate exactly {total_new} new messages as described.\n"
    )

    chat = client.chat.create(model=model, temperature=temperature)
    chat.append(system(prompt))
    response = chat.sample()
    content = response.content
    repaired = repair_json(content)
    if repaired:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, list):
                new_conv = parsed
            else:
                new_conv = parsed.get("conversations", [])
        except:
            new_conv = []
    else:
        try:
            new_conv = extract_json_array(content)
        except ValueError:
            raise ValueError("Could not parse extension")

    if not isinstance(new_conv, list):
        raise ValueError("Extension response missing conversations list")

    if len(new_conv) != total_new:
        raise ValueError(
            f"Extension returned {len(new_conv)} messages, expected {total_new}"
        )

    ok, err = validate_conversations(new_conv)
    if not ok:
        raise ValueError(f"Invalid new conversations: {err}")

    extended_entry = {
        "conversations": conv + new_conv
    }

    ok, err = validate_sharegpt_entry(extended_entry)
    if not ok:
        raise ValueError(f"Extended entry is invalid: {err}")

    return extended_entry


def improve_and_extend_entry(
    entry,
    client,
    model,
    system_prompt,
    num_pairs,
    example_entries=None,
    temperature: float | None = None,
):

    if num_pairs <= 0:
        return improve_entry(entry, client, model, system_prompt)

    ok, err = validate_sharegpt_entry(entry)
    if not ok:
        raise ValueError(f"Cannot process invalid ShareGPT entry: {err}")
    conv = entry.get("conversations", [])
    if not conv or conv[-1].get("from") != "gpt":
        last_role = conv[-1].get("from") if conv else "none"
        raise ValueError(f"Cannot extend conversation ending in '{last_role}', expected 'gpt'")

    total_new = num_pairs * 2

    prefix = _with_system_prefix(system_prompt)

    if example_entries is None:
        example_entries = []
    example_entries = example_entries[:25]

    simple_examples = [ex["conversations"] for ex in example_entries]

    examples_str = ""
    if simple_examples:
        examples_str = "Here are some example conversations for style reference:\n" + "\n\n---\n\n".join(
            json.dumps(s_ex, ensure_ascii=False, indent=2) for s_ex in simple_examples
        ) + "\n\n"

    combined_instructions = (
        "You are improving and extending a ShareGPT-style conversation in one step.\n"
        "The conversations is a list of messages.\n"
        "Each message: dict with 'from' field ('system'|'human'|'gpt') and 'value' field (str)\n\n"
        "CRITICAL SCHEMA RULES - MUST FOLLOW EXACTLY:\n"
        "  1) Rewrite ALL existing 'value' strings for higher quality, coherence, style consistency, and diversity (vary human phrasing and topics to avoid repetition).\n"
        "     Do NOT change 'from' fields, order, or number of existing messages.\n"
        "  2) After rewriting, append exactly {total_new} NEW messages to CONTINUE the conversation.\n"
        "  3) Existing ends with 'gpt'; first new must be 'human', then alternate human/gpt, end with 'gpt'.\n"
        "  4) Optional leading 'system' in existing; preserve it.\n"
        "  5) After system, roles alternate human/gpt; final overall must end on 'gpt'.\n"
        "  6) Output the FULL conversations array as JSON array (rewritten existing + new).\n"
        "  7) Valid JSON array only, no markdown, no extra text. Double-check schema compliance.\n\n"
        "Violations will cause rejection. Ensure exact structure and role sequence.\n"
    ).format(total_new=total_new)

    prompt = (
        prefix
        + combined_instructions
        + examples_str
        + "Original conversations to rewrite and extend:\n"
        + json.dumps(entry["conversations"], ensure_ascii=False, indent=2)
        + f"\n\nRewrite existing and append {total_new} new messages as described.\n"
    )

    chat = client.chat.create(model=model, temperature=temperature)
    chat.append(system(prompt))
    response = chat.sample()
    content = response.content
    repaired = repair_json(content)
    if repaired:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, list):
                combined_conv = parsed
            else:
                combined_conv = parsed.get("conversations", [])
        except:
            combined_conv = []
    else:
        try:
            combined_conv = extract_json_array(content)
        except ValueError:
            raise ValueError("Could not parse combined")

    ok, err = validate_conversations(combined_conv)
    if not ok:
        # Fallback: try improve then extend separately
        try:
            improved = improve_entry(entry, client, model, system_prompt, temperature)
            return extend_entry(improved, client, model, system_prompt, num_pairs, example_entries, temperature)
        except Exception as fallback_error:
            # If improve_entry fails, try to fix the combined_conv directly
            fixed_entry = fix_sharegpt_entry({"conversations": combined_conv})
            if fixed_entry is not None:
                fixed_conv = fixed_entry["conversations"]
                ok2, err2 = validate_conversations(fixed_conv)
                if ok2:
                    return {"conversations": fixed_conv}
            # If all else fails, raise with context
            raise ValueError(
                f"Could not improve and extend entry. Combined validation error: {err}. "
                f"Fallback improve_entry error: {fallback_error}"
            )

    if len(combined_conv) < len(entry["conversations"]):
        raise ValueError("Combined conversations too short")

    return {"conversations": combined_conv}


def generate_human_turns(
    client,
    model,
    system_prompt,
    num_turns: int = 10,
    temperature: float = 0.7,
):
    # Improved generation with smaller batches and stricter natural language enforcement

    prefix = _with_system_prefix(system_prompt)

    valid_domains = ["Adult", "Arts & Entertainment", "Autos & Vehicles", "Beauty & Fitness", "Books & Literature", "Business & Industrial", "Computers & Electronics", "Finance", "Food & Drink", "Games", "Health", "Hobbies & Leisure", "Home & Garden", "Internet & Telecom", "Jobs & Education", "Law & Government", "Online Communities", "People & Society", "Pets & Animals", "Real Estate", "Science", "Sensitive Subjects", "Shopping", "Sports", "Travel & Transportation"]
    num_domains = len(valid_domains)
    target = num_turns
    human_turns = []
    seen = set()
    batch_size = 50  # Smaller batches for better quality control
    max_attempts = 5

    for attempt in range(max_attempts):
        if len(human_turns) >= target:
            break
        current_batch_size = min(batch_size, target - len(human_turns))

        num_per_domain = current_batch_size // num_domains
        instructions = (
            f"CRITICAL: Generate EXACTLY {current_batch_size} FULL CONVERSATIONAL PARAGRAPHS (100-300 words each) that real humans would type to an AI.\n"
            f"Distribute across these {num_domains} domains: {', '.join(valid_domains)} (roughly {num_per_domain} per domain).\n\n"
            "ABSOLUTE RULES - VIOLATIONS WILL BE REJECTED:\n"
            "1. ZERO KEYWORDS/LISTS. Full flowing paragraphs ONLY.\n"
            "2. NO ``` markdown, NO bullet points, NO abbreviations.\n"
            "3. Each message = 1 complete user query (100-300 words).\n"
            "4. Real user frustration/curiosity/problems. Casual natural language.\n"
            "5. COMPLETELY UNIQUE - no similar phrasing across messages.\n\n"
            "EXAMPLE (DO NOT COPY - generate similar quality):\n"
            "[{\"domain\": \"Computers & Electronics\", \"message\": \"Hey, I've been fighting this React bug for hours and it's driving me insane... [continues with detailed problem description]\"}]\n\n"
            "Output ONLY valid JSON array. NO explanations. NO markdown:\n"
        )

        base_prompt = prefix + instructions

        chat = client.chat.create(model=model, temperature=temperature)
        chat.append(system(base_prompt))
        response = chat.sample()
        content = response.content

        repaired = repair_json(content)
        batch_turns = []
        if repaired:
            try:
                parsed = json.loads(repaired)
                print(f"DEBUG llm_helpers generate_human_turns batch {attempt}: parsed type={type(parsed)}, len={len(parsed) if hasattr(parsed, '__len__') else 'N/A'}")
                if isinstance(parsed, list):
                    batch_turns = parsed
                else:
                    batch_turns = parsed.get("human_turns", [])
            except Exception as parse_e:
                print(f"DEBUG llm_helpers parse error batch {attempt}: {parse_e}")
                batch_turns = []
        else:
            try:
                batch_turns = extract_json_array(content)
                print(f"DEBUG llm_helpers extract_json_array batch {attempt}: len={len(batch_turns)}")
            except ValueError as e:
                print(f"DEBUG llm_helpers extract error batch {attempt}: {e}")

        batch_valid = []
        for item in batch_turns:
            if isinstance(item, dict) and 'message' in item:
                msg = item['message'].strip()
                if len(msg) < 100 or len(msg) > 500 or '```' in msg or '\n\n' in msg:  # Reject short/keyword/markdown dumps
                    continue
                raw_domain = item.get('domain')
                domain = _normalize_domain(raw_domain, valid_domains)
                if domain is None or domain == "Other":
                    continue
                key = (domain, msg.lower())
                if key not in seen:
                    batch_valid.append({'domain': domain, 'message': msg})
                    seen.add(key)
        human_turns.extend(batch_valid)
        print(f"DEBUG llm_helpers human_turns batch {attempt}: added {len(batch_valid)}, total {len(human_turns)}")

    print(f"DEBUG llm_helpers generate_human_turns final: {len(human_turns)} / {target}")

    if len(human_turns) < max(1, target // 4):
        print(f"WARNING: Low human turn yield {len(human_turns)}/{target}, returning partial.")

    return human_turns[:num_turns]


def rewrite_human_cache(
    client,
    model,
    system_prompt,
    cache_file="global_human_cache.json",
    max_entries=None,
    temperature=0.8,
    q=None  # Add queue for GUI logging
):
    """Rewrite existing human turns cache into proper conversational ShareGPT entries"""
    import os
    import time
    
    if not os.path.exists(cache_file):
        print(f"No cache file found at {cache_file}")
        return []
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            raw_cache = json.load(f)
    except Exception as e:
        print(f"Failed to load cache {cache_file}: {e}")
        return []
    
    rewritten_cache = []
    valid_domains = ["Adult", "Arts & Entertainment", "Autos & Vehicles", "Beauty & Fitness", "Books & Literature", "Business & Industrial", "Computers & Electronics", "Finance", "Food & Drink", "Games", "Health", "Hobbies & Leisure", "Home & Garden", "Internet & Telecom", "Jobs & Education", "Law & Government", "Online Communities", "People & Society", "Pets & Animals", "Real Estate", "Science", "Sensitive Subjects", "Shopping", "Sports", "Travel & Transportation"]
    
    if max_entries is None:
        max_entries = len(raw_cache)
        
    batch_size = 10  # Process in small aggressive batches
    max_retries = 10
    
    if q:
        q.put(("log", f"🚀 Starting aggressive rewrite of {len(raw_cache)} cache entries in {len(raw_cache)//batch_size}+ batches"))
    else:
        print(f"Starting aggressive rewrite of {len(raw_cache[:max_entries])} cache entries in batches of {batch_size}")
    
    for batch_start in range(0, len(raw_cache), batch_size):
        batch_end = min(batch_start + batch_size, len(raw_cache))
        batch_items = raw_cache[batch_start:batch_end]
        batch_rewritten = []
        
        batch_num = batch_start//batch_size + 1
        if q:
            q.put(("log", f"📦 Batch {batch_num}: processing entries {batch_start+1}-{batch_end}"))
        else:
            print(f"Processing batch {batch_start//batch_size + 1}: entries {batch_start+1}-{batch_end}")
        
        for i, item in enumerate(batch_items):
            global_i = batch_start + i
            if global_i >= max_entries:
                break
                
            if not isinstance(item, dict) or 'message' not in item:
                continue
                
            human_msg = item['message'].strip()
            if len(human_msg) < 50:
                continue
                
            domain = item.get('domain', 'Other')
            if domain not in valid_domains:
                continue
                
            rewritten = False
            for retry in range(max_retries):
                try:
                    if q and retry > 0:
                        q.put(("log", f"🔄 RETRY {retry+1}/{max_retries} entry {global_i+1} ({domain})"))
                    
                    conv = [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": human_msg},
                        {"from": "gpt", "value": f"I understand your question about {domain.lower()} and will provide a detailed, helpful response."}
                    ]
                    entry = {"conversations": conv}
                    
                    if q:
                        q.put(("log", f"🔄 API CALL entry {global_i+1} ({domain})"))
                    
                    improved_entry = improve_entry(entry, client, model, system_prompt, temperature)
                    
                    if q:
                        q.put(("log", f"✅ API entry {global_i+1} ({domain})"))
                    
                    # Extract rewritten human message
                    for msg in improved_entry['conversations']:
                        if msg['from'] == 'human':
                            batch_rewritten.append({
                                'domain': domain,
                                'message': msg['value']
                            })
                            if q:
                                q.put(("log", f"✓ Rewrote entry {global_i+1}: {domain}"))
                            rewritten = True
                    break
                    
                except Exception as e:
                    if retry == max_retries - 1:
                        if q:
                            q.put(("log", f"✗ FAILED entry {global_i+1} after {max_retries} retries"))
                        else:
                            print(f"✗ FAILED entry {global_i+1} after {max_retries} retries: {str(e)[:100]}")
                    else:
                        time.sleep(0.1)  # Brief pause between retries
                    continue
            
            if not rewritten:
                # Fallback: keep original if all retries fail
                batch_rewritten.append(item)
        
        rewritten_cache.extend(batch_rewritten)
        if q:
            q.put(("log", f"✅ Batch {batch_num} complete: {len(batch_rewritten)}/{len(batch_items)} items"))
        else:
            print(f"Batch {batch_start//batch_size + 1} complete: {len(batch_rewritten)} items")
    
    if q:
        q.put(("log", f"🎉 FINAL: Rewrote {len(rewritten_cache)}/{max_entries} cache entries COMPLETE"))
    else:
        print(f"Final rewritten cache size: {len(rewritten_cache)}")
    
    # Save rewritten cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(rewritten_cache, f, indent=2, ensure_ascii=False)
        success_msg = f"Successfully rewrote {len(rewritten_cache)} / {max_entries} cache entries to {cache_file}"
        if q:
            q.put(("log", success_msg))
        else:
            print(success_msg)
    except Exception as e:
        print(f"Failed to save rewritten cache: {e}")
    
    return rewritten_cache
    
    return rewritten_cache


def generate_single_gpt_response(
    client,
    model,
    system_prompt,
    conversations,
    temperature: float | None = None,
) -> str:

    prefix = _with_system_prefix(system_prompt)
    instructions = (
        "You are the gpt AI. Respond naturally to the last human message in this conversation.\n\n"
        "Output ONLY the raw text of your gpt response. "
        "No JSON, no markdown, no code fences, no explanations, no lists or bullets. "
        "Just the direct, coherent continuation."
    )
    prompt = (
        prefix
        + instructions
        + "\n\nCurrent conversation:\n"
        + json.dumps(conversations, ensure_ascii=False, indent=2)
        + "\n\ngpt:"
    )
    chat = client.chat.create(model=model, temperature=temperature)
    chat.append(system(prompt))
    response = chat.sample()
    content = response.content.strip()
    if not content or len(content) < 5:
        return "I understand your point."
    return content


def generate_names(
    client,
    model,
    num_names: int = 20,
    temperature: float = 0.7,
):

    valid_categories = [
        "Female", "Male", "Non-binary", "Senior",
        "Fantasy", "Sci-Fi", "Historical", "Steampunk", "Cyberpunk",
        "Warrior", "Mage", "Merchant", "Noble", "Peasant",
        "Scientist", "Artist", "Musician", "Athlete", "Villain", "Hero"
    ]
    num_categories = len(valid_categories)
    target = num_names
    names_list = []
    seen = set()
    batch_size = 1000
    max_attempts = 5
    attempt = 0
    while len(names_list) < target and attempt < max_attempts:
        current_batch_size = min(batch_size, target - len(names_list))

        instructions = (
            "All characters must be adults (18 years or older). Avoid any names or themes implying children, minors, or anyone under 18. Focus on mature, adult personas.\n\n"
            f"You are generating diverse, unique FULL NAMES for characters inspired by {num_categories} categories: {', '.join(valid_categories)}.\n"
            f"Distribute roughly evenly, generate EXACTLY {current_batch_size} COMPLETELY UNIQUE names.\n"
            "Names culturally/ethnically diverse, realistic or fitting category/theme (e.g., 'Elara Stormwind' for Fantasy).\n"
            "CRITICAL: ZERO duplicates/similar names. Every 'name' distinctly different.\n"
            "Vary styles: common, exotic, compound, nicknames where fitting.\n\n"
            "Output MUST be valid JSON array of objects, exactly {current_batch_size} items.\n"
            "Each: {{\"category\": \"one of the categories above\", \"name\": \"full unique name string\"}}\n"
            "No markdown, explanations, ONLY JSON array.\n"
            "Example: [{{\"category\": \"Female\", \"name\": \"Alice Johnson\"}}, {{\"category\": \"Warrior\", \"name\": \"Kragthar Bloodaxe\"}}, ...]\n"
        )

        base_prompt = instructions

        chat = client.chat.create(model=model, temperature=temperature)
        chat.append(system(base_prompt))
        response = chat.sample()
        content = response.content

        repaired = repair_json(content)
        batch_names = []
        if repaired:
            try:
                parsed = json.loads(repaired)
                print(f"DEBUG llm_helpers generate_names batch {attempt}: parsed type={type(parsed)}, len={len(parsed) if hasattr(parsed, '__len__') else 'N/A'}")
                if isinstance(parsed, list):
                    batch_names = parsed
                else:
                    batch_names = parsed.get("names", [])
            except Exception as parse_e:
                print(f"DEBUG llm_helpers names parse error batch {attempt}: {parse_e}")
                batch_names = []
        else:
            try:
                batch_names = extract_json_array(content)
                print(f"DEBUG llm_helpers names extract batch {attempt}: len={len(batch_names)}")
            except ValueError as e:
                print(f"DEBUG llm_helpers names extract error batch {attempt}: {e}")

        batch_valid = []
        for item in batch_names:
            if isinstance(item, dict) and 'name' in item and item['name'].strip():
                name = item['name'].strip()
                if name.lower() not in seen:
                    category = item.get('category', 'Generic').title()
                    batch_valid.append({'category': category, 'name': name})
                    seen.add(name.lower())
        names_list.extend(batch_valid)
        print(f"DEBUG llm_helpers names batch {attempt}: added {len(batch_valid)}, total {len(names_list)}")
        attempt += 1

    print(f"DEBUG llm_helpers generate_names final: {len(names_list)} / {target}")

    if len(names_list) < target * 0.5:
        print(f"WARNING: Low name yield {len(names_list)}/{target}, but proceeding with available.")

    return names_list
