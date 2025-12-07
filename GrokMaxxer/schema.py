import json
import re

try:
    from .model_config import ALLOWED_ROLES, SCHEMA_DESCRIPTION
except ImportError:
    # Fallback for when run as a script - add GrokMaxxer to path
    import sys
    from pathlib import Path
    grokmaxxer_dir = Path(__file__).parent
    if str(grokmaxxer_dir) not in sys.path:
        sys.path.insert(0, str(grokmaxxer_dir))
    from model_config import ALLOWED_ROLES, SCHEMA_DESCRIPTION


def extract_json_array(text):

    if not isinstance(text, str):
        raise ValueError("Input must be string")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except:
        pass
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

    if not isinstance(text, str):
        return text

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("No opening brace found in response")
    
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
        except Exception:
            pass
        raise ValueError("Unbalanced braces in JSON response")
    
    chunk = text[start_idx:end_idx]
    try:
        return json.loads(chunk)
    except Exception:

        try:
            from json_repair import repair_json
            repaired = repair_json(chunk)
            if repaired:
                return json.loads(repaired)
        except Exception:
            pass
        raise ValueError("Could not parse JSON from response")


def validate_sharegpt_entry(entry, lenient=False):

    if not isinstance(entry, dict):
        return False, "Entry is not a JSON object"

    conv = entry.get("conversations")
    if not isinstance(conv, list) or not conv:
        return False, "Missing or empty 'conversations' list"

    for i, msg in enumerate(conv):
        if not isinstance(msg, dict):
            return False, f"Message {i} is not an object"
        role = msg.get("from")
        value = msg.get("value")
        if role not in ALLOWED_ROLES:
            if lenient:

                continue
            return False, f"Invalid role '{role}' at index {i}"
        if not isinstance(value, str):
            if lenient:

                continue
            return False, f"'value' at index {i} is not a string"

    idx = 0
    if conv[0]["from"] == "system":
        idx = 1

    if idx >= len(conv):
        return False, "Conversation has only a system message and nothing else"

    if conv[idx]["from"] != "human":
        if lenient:

            pass
        else:
            return False, f"First non-system message must be 'human', got '{conv[idx]['from']}'"

    expected = "human"
    alternation_errors = []
    for j in range(idx, len(conv)):
        role = conv[j]["from"]
        if role != expected:
            if lenient:
                alternation_errors.append(f"Expected '{expected}' at index {j}, got '{role}'")
            else:
                return False, f"Expected '{expected}' at index {j}, got '{role}'"
        expected = "gpt" if expected == "human" else "human"

    if conv[-1]["from"] == "human":
        if lenient:

            pass
        else:
            return False, "Conversation ends on 'human'"

    return True, None


def validate_conversations(conv, lenient=False):

    if not isinstance(conv, list) or not conv:
        return False, "Empty conversations list"
    for i, msg in enumerate(conv):
        if not isinstance(msg, dict):
            return False, f"Message {i} not dict"
        role = msg.get("from")
        value = msg.get("value")
        if role not in ALLOWED_ROLES:
            if lenient:
                continue
            return False, f"Invalid role '{role}' at {i}"
        if not isinstance(value, str):
            if lenient:
                continue
            return False, f"Value at {i} not str"
    idx = 0
    if conv and conv[0].get("from") == "system":
        idx = 1
    if idx >= len(conv):
        return False, "Only system message"
    if conv[idx].get("from") != "human":
        if lenient:
            pass
        else:
            return False, f"First non-system not human: {conv[idx].get('from')}"
    expected = "human"
    for j in range(idx, len(conv)):
        role = conv[j].get("from")
        if role != expected:
            if lenient:
                pass
            else:
                return False, f"Expected {expected} at {j}, got {role}"
        expected = "gpt" if expected == "human" else "human"
    if conv[-1].get("from") == "human":
        if lenient:
            pass
        else:
            return False, "Ends on human"
    return True, None


def ensure_system_message(entry, system_prompt):

    if not system_prompt:
        return entry
    if not isinstance(entry, dict):
        return entry

    conv = entry.get("conversations")
    if not isinstance(conv, list) or not conv:
        return entry

    if conv[0].get("from") == "system":
        conv[0]["value"] = system_prompt
    else:
        conv.insert(0, {"from": "system", "value": system_prompt})

    return entry


def fix_sharegpt_entry(entry):

    if not isinstance(entry, dict):
        return None

    conv = entry.get("conversations")
    if not isinstance(conv, list):
        conv = []
        entry["conversations"] = conv

    if len(conv) < 1:
        return None

    fixed_conv = []
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        if "from" not in msg or "value" not in msg:
            continue
        
        value = msg.get("value")
        if not isinstance(value, str):
            if value is None:
                value = " "
            else:
                value = str(value)
            msg["value"] = value
        
        role = msg.get("from")
        if role not in ALLOWED_ROLES:

            role_lower = str(role).lower() if role else ""
            if "user" in role_lower or "human" in role_lower or "person" in role_lower:
                msg["from"] = "human"
            elif "assistant" in role_lower or "ai" in role_lower or "bot" in role_lower or "model" in role_lower:
                msg["from"] = "gpt"
            elif "system" in role_lower:
                msg["from"] = "system"
            else:

                msg["from"] = "gpt"
        
        fixed_conv.append(msg)
    
    conv = fixed_conv
    if len(conv) < 2:
        return None

    system_msgs = [i for i, msg in enumerate(conv) if msg["from"] == "system"]
    if len(system_msgs) > 1:

        for i in sorted(system_msgs[1:], reverse=True):
            del conv[i]
    elif system_msgs:

        if system_msgs[0] != 0:
            conv.insert(0, conv.pop(system_msgs[0]))

    idx = 1 if conv and conv[0]["from"] == "system" else 0

    if idx >= len(conv):
        return None

    if conv[idx]["from"] != "human":
        conv[idx]["from"] = "human"

    expected = "human"
    for j in range(idx, len(conv)):
        role = conv[j]["from"]
        if role != expected:

            conv[j]["from"] = expected
        expected = "gpt" if expected == "human" else "human"

    for msg in conv:
        value = msg.get("value", "")
        if not isinstance(value, str) or not value.strip():
            msg["value"] = " "

    if conv[-1]["from"] == "human":
        conv.append({"from": "gpt", "value": "I understand."})

    if len(conv) < 2:
        return None

    return entry
