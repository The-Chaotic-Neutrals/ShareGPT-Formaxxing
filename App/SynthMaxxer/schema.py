"""
Schema utilities for SynthMaxxer
Uses standard chat format internally, converts to ShareGPT on output
"""
import json
import re


def extract_json_array(text):
    """Extract a JSON array from text, handling markdown code blocks"""
    if not isinstance(text, str):
        raise ValueError("Input must be string")
    text = text.strip()
    
    # Strip markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()
    
    try:
        return json.loads(text)
    except:
        pass
    
    # Find array bounds
    start_idx = text.find("[")
    if start_idx == -1:
        raise ValueError("No opening [ found")
    
    bracket_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        if text[i] == "[":
            bracket_count += 1
        elif text[i] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i + 1
                break
    
    if bracket_count != 0:
        try:
            from json_repair import repair_json
            repaired = repair_json(text)
            if repaired:
                return json.loads(repaired)
        except:
            pass
        raise ValueError("Unbalanced brackets")
    
    chunk = text[start_idx:end_idx]
    try:
        return json.loads(chunk)
    except:
        try:
            from json_repair import repair_json
            repaired = repair_json(chunk)
            if repaired:
                return json.loads(repaired)
        except:
            pass
        raise ValueError("Could not parse array")


def extract_json_object(text):
    """Extract a JSON object from text"""
    if not isinstance(text, str):
        return text

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except:
        pass

    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("No opening brace found")
    
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if brace_count != 0:
        try:
            from json_repair import repair_json
            repaired = repair_json(text)
            if repaired:
                return json.loads(repaired)
        except:
            pass
        raise ValueError("Unbalanced braces")
    
    chunk = text[start_idx:end_idx]
    try:
        return json.loads(chunk)
    except:
        try:
            from json_repair import repair_json
            repaired = repair_json(chunk)
            if repaired:
                return json.loads(repaired)
        except:
            pass
        raise ValueError("Could not parse JSON")


# ============================================================================
# Standard Chat Format (internal)
# ============================================================================
# [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "..."},
# ]

def validate_messages(messages):
    """Validate a list of chat messages"""
    if not isinstance(messages, list) or not messages:
        return False, "Empty messages list"
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message {i} is not a dict"
        if "role" not in msg or "content" not in msg:
            return False, f"Message {i} missing role or content"
        if msg["role"] not in ("system", "user", "assistant"):
            return False, f"Invalid role at {i}: {msg['role']}"
        if not isinstance(msg["content"], str):
            return False, f"Content at {i} is not a string"
    
    # Check structure: optional system, then user/assistant alternation
    idx = 0
    if messages[0]["role"] == "system":
        idx = 1
    
    if idx >= len(messages):
        return False, "Only system message present"
    
    if messages[idx]["role"] != "user":
        return False, f"First non-system must be user, got {messages[idx]['role']}"
    
    # Check alternation
    expected = "user"
    for j in range(idx, len(messages)):
        if messages[j]["role"] != expected:
            return False, f"Expected {expected} at {j}, got {messages[j]['role']}"
        expected = "assistant" if expected == "user" else "user"
    
    if messages[-1]["role"] == "user":
        return False, "Conversation ends on user"
    
    return True, None


def fix_messages(messages):
    """Fix common issues in message lists"""
    if not isinstance(messages, list) or not messages:
        return None
    
    fixed = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        # Normalize role names
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))
        
        if not role or not isinstance(content, str):
            continue
        
        role = str(role).lower()
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "ai", "bot", "model"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "assistant"  # Default unknown to assistant
        
        fixed.append({"role": role, "content": content.strip() or " "})
    
    if len(fixed) < 2:
        return None
    
    # Ensure system is first if present
    system_msgs = [i for i, m in enumerate(fixed) if m["role"] == "system"]
    if len(system_msgs) > 1:
        # Keep only first system message
        for i in sorted(system_msgs[1:], reverse=True):
            del fixed[i]
    elif system_msgs and system_msgs[0] != 0:
        fixed.insert(0, fixed.pop(system_msgs[0]))
    
    # Fix alternation
    idx = 1 if fixed and fixed[0]["role"] == "system" else 0
    if idx >= len(fixed):
        return None
    
    # First non-system must be user
    if fixed[idx]["role"] != "user":
        fixed[idx]["role"] = "user"
    
    expected = "user"
    for j in range(idx, len(fixed)):
        if fixed[j]["role"] != expected:
            fixed[j]["role"] = expected
        expected = "assistant" if expected == "user" else "user"
    
    # Must end on assistant
    if fixed[-1]["role"] == "user":
        fixed.append({"role": "assistant", "content": "I understand."})
    
    return fixed if len(fixed) >= 2 else None


def add_system_message(messages, system_prompt):
    """Add or update system message at the start"""
    if not system_prompt or not isinstance(messages, list):
        return messages
    
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system_prompt
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    return messages


# ============================================================================
# ShareGPT Format Conversion (for output)
# ============================================================================

def to_sharegpt(messages):
    """Convert standard messages to ShareGPT format"""
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    return {
        "conversations": [
            {"from": role_map.get(m["role"], m["role"]), "value": m["content"]}
            for m in messages
        ]
    }


def from_sharegpt(entry):
    """Convert ShareGPT format to standard messages"""
    if not isinstance(entry, dict):
        return None
    
    conv = entry.get("conversations", [])
    if not isinstance(conv, list):
        return None
    
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role and isinstance(content, str):
            messages.append({
                "role": role_map.get(role, role),
                "content": content
            })
    
    return messages if messages else None


# ============================================================================
# Legacy compatibility (for existing code that uses old function names)
# ============================================================================

def validate_sharegpt_entry(entry, lenient=False):
    """Validate a ShareGPT format entry"""
    messages = from_sharegpt(entry)
    if not messages:
        return False, "Invalid ShareGPT entry"
    return validate_messages(messages)


def validate_conversations(conv, lenient=False):
    """Validate a conversations list (ShareGPT format)"""
    messages = from_sharegpt({"conversations": conv})
    if not messages:
        return False, "Invalid conversations"
    return validate_messages(messages)


def fix_sharegpt_entry(entry):
    """Fix a ShareGPT format entry"""
    messages = from_sharegpt(entry)
    if not messages:
        return None
    fixed = fix_messages(messages)
    if not fixed:
        return None
    return to_sharegpt(fixed)


def ensure_system_message(entry, system_prompt):
    """Add system message to ShareGPT entry"""
    messages = from_sharegpt(entry)
    if not messages:
        return entry
    messages = add_system_message(messages, system_prompt)
    return to_sharegpt(messages)


# Keep for backward compatibility
ALLOWED_ROLES = {"system", "human", "gpt"}
SCHEMA_DESCRIPTION = "Standard chat format with system/user/assistant roles"
