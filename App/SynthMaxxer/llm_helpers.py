"""
LLM helper functions for SynthMaxxer
Uses standard chat format internally, converts to ShareGPT on output
"""
import json
from json_repair import repair_json

import sys
from pathlib import Path
_synthmaxxer_dir = Path(__file__).parent
_parent_dir = _synthmaxxer_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from App.SynthMaxxer.schema import (
    extract_json_array,
    validate_messages,
    fix_messages,
    to_sharegpt,
    from_sharegpt,
    add_system_message,
)
from App.SynthMaxxer.llm_client import LLMClient


def _parse_messages_response(content):
    """Parse LLM response into messages list"""
    repaired = repair_json(content) or content
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Handle {"messages": [...]} or {"conversations": [...]}
            return parsed.get("messages", parsed.get("conversations", []))
    except:
        pass
    
    try:
        return extract_json_array(content)
    except:
        return []


def improve_entry(entry, client: LLMClient, model: str, system_prompt: str, temperature: float = 0.7):
    """Improve an existing conversation by rewriting the content"""
    messages = from_sharegpt(entry)
    if not messages:
        raise ValueError("Could not parse entry")
    
    prompt = (
        "Rewrite this conversation to improve quality, clarity, and naturalness.\n\n"
        "RULES:\n"
        "- Keep the same number of messages\n"
        "- Keep the same roles in the same order\n"
        "- Only improve the content/wording\n"
        "- Output as JSON array: [{\"role\": \"...\", \"content\": \"...\"}, ...]\n\n"
        f"Original:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
        "Output ONLY the JSON array, no markdown or explanation."
    )
    
    if system_prompt:
        prompt = system_prompt.strip() + "\n\n" + prompt
    
    content = client.chat_completion(
        model=model,
        system_prompt="You are a helpful assistant that improves conversation data.",
        user_message=prompt,
        temperature=temperature
    )
    
    improved = _parse_messages_response(content)
    if len(improved) != len(messages):
        raise ValueError(f"Length mismatch: expected {len(messages)}, got {len(improved)}")
    
    fixed = fix_messages(improved)
    if not fixed:
        raise ValueError("Could not fix improved messages")
    
    return to_sharegpt(fixed)


def generate_new_entry(client: LLMClient, model: str, system_prompt: str, example_entries=None, temperature: float = 0.7):
    """Generate a new conversation"""
    examples_str = ""
    if example_entries:
        examples = []
        for ex in example_entries[:3]:
            msgs = from_sharegpt(ex)
            if msgs:
                examples.append(msgs)
        if examples:
            examples_str = "Example conversations:\n" + "\n---\n".join(
                json.dumps(ex, ensure_ascii=False) for ex in examples
            ) + "\n\n"
    
    prompt = (
        "Generate a new conversation between a user and an assistant.\n\n"
        f"{examples_str}"
        "RULES:\n"
        "- Optional system message first\n"
        "- Then alternating: user, assistant, user, assistant, ...\n"
        "- End with assistant\n"
        "- At least 2-4 exchanges\n"
        "- Output as JSON array: [{\"role\": \"...\", \"content\": \"...\"}, ...]\n\n"
        "Output ONLY the JSON array, no markdown or explanation."
    )
    
    if system_prompt:
        prompt = system_prompt.strip() + "\n\n" + prompt
    
    content = client.chat_completion(
        model=model,
        system_prompt="You are a helpful assistant that generates conversation data.",
        user_message=prompt,
        temperature=temperature
    )
    
    messages = _parse_messages_response(content)
    fixed = fix_messages(messages)
    if not fixed:
        raise ValueError("Could not parse/fix generated conversation")
    
    return to_sharegpt(fixed)


def extend_entry(entry, client: LLMClient, model: str, system_prompt: str, num_pairs: int, example_entries=None, temperature: float = 0.7):
    """Extend a conversation with additional exchanges"""
    if num_pairs <= 0:
        return entry
    
    messages = from_sharegpt(entry)
    if not messages:
        raise ValueError("Could not parse entry")
    
    if messages[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with assistant to extend")
    
    prompt = (
        f"Continue this conversation with {num_pairs} more exchange(s) (user then assistant).\n\n"
        f"Current conversation:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
        "RULES:\n"
        "- Generate ONLY the new messages (not the existing ones)\n"
        "- Start with user, end with assistant\n"
        f"- Generate exactly {num_pairs * 2} new messages\n"
        "- Output as JSON array: [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}, ...]\n\n"
        "Output ONLY the JSON array of NEW messages."
    )
    
    if system_prompt:
        prompt = system_prompt.strip() + "\n\n" + prompt
    
    content = client.chat_completion(
        model=model,
        system_prompt="You are a helpful assistant that extends conversations.",
        user_message=prompt,
        temperature=temperature
    )
    
    new_messages = _parse_messages_response(content)
    if len(new_messages) != num_pairs * 2:
        raise ValueError(f"Expected {num_pairs * 2} new messages, got {len(new_messages)}")
    
    combined = messages + new_messages
    fixed = fix_messages(combined)
    if not fixed:
        raise ValueError("Could not fix extended conversation")
    
    return to_sharegpt(fixed)


def improve_and_extend_entry(entry, client: LLMClient, model: str, system_prompt: str, num_pairs: int, example_entries=None, temperature: float = 0.7):
    """Improve and extend a conversation in one operation"""
    if num_pairs <= 0:
        return improve_entry(entry, client, model, system_prompt, temperature)
    
    messages = from_sharegpt(entry)
    if not messages:
        raise ValueError("Could not parse entry")
    
    prompt = (
        f"Improve this conversation AND add {num_pairs} more exchange(s).\n\n"
        f"Current:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
        "RULES:\n"
        "- Rewrite existing messages for better quality\n"
        "- Keep same structure for existing messages\n"
        f"- Add {num_pairs * 2} new messages at the end (user, assistant, ...)\n"
        "- End with assistant\n"
        "- Output the FULL conversation as JSON array\n\n"
        "Output ONLY the JSON array."
    )
    
    if system_prompt:
        prompt = system_prompt.strip() + "\n\n" + prompt
    
    content = client.chat_completion(
        model=model,
        system_prompt="You are a helpful assistant that improves and extends conversations.",
        user_message=prompt,
        temperature=temperature
    )
    
    combined = _parse_messages_response(content)
    fixed = fix_messages(combined)
    
    if not fixed:
        # Fallback: try separate operations
        try:
            improved = improve_entry(entry, client, model, system_prompt, temperature)
            return extend_entry(improved, client, model, system_prompt, num_pairs, example_entries, temperature)
        except:
            raise ValueError("Could not improve and extend")
    
    return to_sharegpt(fixed)


def generate_single_response(client: LLMClient, model: str, system_prompt: str, messages, temperature: float = 0.7) -> str:
    """Generate a single assistant response for a conversation"""
    prompt = (
        "Continue this conversation with ONE assistant response.\n\n"
        f"Conversation:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
        "Output ONLY the assistant's response text (no JSON, no quotes)."
    )
    
    if system_prompt:
        prompt = system_prompt.strip() + "\n\n" + prompt
    
    content = client.chat_completion(
        model=model,
        system_prompt="You are a helpful assistant.",
        user_message=prompt,
        temperature=temperature
    )
    
    return content.strip() or "I understand."


def generate_names(client: LLMClient, model: str, num_names: int = 20, temperature: float = 0.7):
    """Generate diverse character names with ages (18+)"""
    names_list = []
    seen = set()
    batch_size = min(100, num_names)
    
    for _ in range(5):  # Max attempts
        if len(names_list) >= num_names:
            break
        
        need = min(batch_size, num_names - len(names_list))
        
        prompt = (
            f"Generate {need} unique character names with ages.\n\n"
            "Requirements:\n"
            "- All must be adults (18+)\n"
            "- Diverse cultures/ethnicities\n"
            "- Each name unique\n\n"
            "Output ONLY JSON array: [{\"name\": \"Full Name\", \"age\": 25}, ...]"
        )
        
        content = client.chat_completion(
            model=model,
            system_prompt="You generate character names.",
            user_message=prompt,
            temperature=temperature
        )
        
        try:
            batch = _parse_messages_response(content)
            for item in batch:
                if isinstance(item, dict) and item.get("name"):
                    name = str(item["name"]).strip()
                    age = max(18, int(item.get("age", 25))) if isinstance(item.get("age"), (int, float)) else 25
                    if name.lower() not in seen:
                        names_list.append({"name": name, "age": age})
                        seen.add(name.lower())
        except:
            pass
    
    return names_list[:num_names]
