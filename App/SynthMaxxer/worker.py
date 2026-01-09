"""
Worker function for SynthMaxxer - handles the generation loop
Can be called from GUI or CLI
"""
import json
import os
import random
import re
import requests
import sys
import time
import pathlib
from datetime import datetime


def worker(
    api_key,
    endpoint,
    model,
    output_dir,
    system_message,
    user_first_message,
    assistant_first_message,
    user_start_tag,
    user_end_tag,
    assistant_start_tag,
    assistant_end_tag,
    is_instruct,
    min_delay,
    max_delay,
    stop_percentage,
    min_turns,
    refusal_phrases,
    force_retry_phrases,
    api_type,
    stop_flag,
    q=None,
):
    """
    Worker function that runs the generation loop
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model: Model name to use
        output_dir: Output directory path (relative to script)
        system_message: System message/prompt
        user_first_message: First user message
        assistant_first_message: First assistant message
        user_start_tag: Tag marking start of user message
        user_end_tag: Tag marking end of user message
        assistant_start_tag: Tag marking start of assistant message
        assistant_end_tag: Tag marking end of assistant message
        is_instruct: Whether in instruct mode
        min_delay: Minimum delay between generations
        max_delay: Maximum delay between generations
        stop_percentage: Probability of stopping after min_turns
        min_turns: Minimum assistant turns before stopping
        refusal_phrases: List of phrases that indicate refusal
        force_retry_phrases: List of phrases that force retry
        api_type: Type of API (Anthropic Claude, OpenAI Official, etc.)
        stop_flag: threading.Event to signal stop
        q: Queue for GUI communication (optional)
    """
    if q:
        q.put(("log", "Worker thread started"))
    
    try:
        # Adjust settings for instruct mode
        actual_min_turns = 0 if is_instruct else min_turns
        start_index = 0 if is_instruct else 2
        actual_stop_percentage = 0.25 if is_instruct else stop_percentage

        # Prepare headers and data based on API type
        if api_type == "Anthropic Claude":
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01"
            }
            assistant_first_with_tags = f"{assistant_start_tag}\n{assistant_first_message}\n\n{assistant_end_tag}\n\n{user_start_tag}"
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": user_first_message
                    },
                    {
                        "role": "assistant",
                        "content": assistant_first_with_tags
                    }
                ],
                "max_tokens": 200000,
                "temperature": 1,
                "top_p": 1,
                "top_k": 0,
                "system": system_message,
                "stream": True
            }
        elif api_type in ["OpenAI Official", "OpenAI Chat Completions", "Grok (xAI)", "OpenRouter", "DeepSeek"]:
            # All OpenAI-compatible chat completions APIs use the same format
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            # OpenRouter needs additional headers
            if api_type == "OpenRouter":
                headers["HTTP-Referer"] = "https://github.com/ShareGPT-Formaxxing"
                headers["X-Title"] = "SynthMaxxer"
            
            assistant_first_with_tags = f"{assistant_start_tag}\n{assistant_first_message}\n\n{assistant_end_tag}\n\n{user_start_tag}"
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.extend([
                {"role": "user", "content": user_first_message},
                {"role": "assistant", "content": assistant_first_with_tags}
            ])
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": 200000,
                "temperature": 1,
                "top_p": 1,
                "stream": True
            }
        elif api_type == "Gemini (Google)":
            # Gemini uses a different format with contents and parts
            if ":streamGenerateContent" not in endpoint:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={api_key}"
            elif "?key=" not in endpoint:
                endpoint = f"{endpoint}?key={api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            assistant_first_with_tags = f"{assistant_start_tag}\n{assistant_first_message}\n\n{assistant_end_tag}\n\n{user_start_tag}"
            
            contents = []
            if system_message:
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {system_message}"}]
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": "Understood."}]
                })
            
            contents.append({
                "role": "user",
                "parts": [{"text": user_first_message}]
            })
            
            contents.append({
                "role": "model",
                "parts": [{"text": assistant_first_with_tags}]
            })
            
            data = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 1,
                    "topP": 1,
                    "maxOutputTokens": 200000
                }
            }
        else:  # OpenAI Text Completions
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            prompt_parts = []
            if system_message:
                prompt_parts.append(f"System: {system_message}\n\n")
            prompt_parts.append(f"User: {user_first_message}\n\n")
            prompt_parts.append(f"Assistant: {assistant_start_tag}\n{assistant_first_message}\n\n{assistant_end_tag}\n\n{user_start_tag}")
            prompt = "".join(prompt_parts)
            data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": 200000,
                "temperature": 1,
                "top_p": 1,
                "stream": True
            }

        # Create output directory
        script_dir = pathlib.Path(__file__).parent.parent.absolute()
        output_path = script_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # Compile patterns
        refusal_pattern = re.compile("|".join(refusal_phrases)) if refusal_phrases else None
        force_retry_pattern = re.compile("|".join(force_retry_phrases)) if force_retry_phrases else None

        # Build prompt metadata to save with each generation
        prompt_metadata = {
            "model": model,
            "endpoint": endpoint,
            "api_type": api_type,
            "system_message": system_message,
            "user_first_message": user_first_message,
            "assistant_first_message": assistant_first_message,
            "user_start_tag": user_start_tag,
            "user_end_tag": user_end_tag,
            "assistant_start_tag": assistant_start_tag,
            "assistant_end_tag": assistant_end_tag,
            "is_instruct": is_instruct,
            "min_turns": actual_min_turns,
            "stop_percentage": actual_stop_percentage,
            "temperature": 1,
            "max_tokens": 200000,
        }

        # Create session
        session = requests.Session()

        if q:
            q.put(("log", f"Starting generation loop..."))
            q.put(("log", f"Output directory: {output_path}"))
            q.put(("log", f"API Type: {api_type}"))

        generation_count = 0

        # For text completions, we need to track the prompt ending
        prompt_ending_for_text = None
        if api_type == "OpenAI Text Completions":
            prompt_ending_for_text = f"{assistant_start_tag}\n{assistant_first_message}\n\n{assistant_end_tag}\n\n{user_start_tag}"

        while not stop_flag.is_set():
            try:
                if q:
                    q.put(("log", f"\n--- Generation #{generation_count + 1} ---"))
                
                with session.post(endpoint, headers=headers, json=data, stream=True) as response:
                    response.raise_for_status()

                    if q:
                        q.put(("log", "Generating response..."))
                    full_response = ""
                    accumulated_content = ""
                    
                    # For OpenAI-compatible APIs, we need to track the full conversation including the initial assistant message
                    if api_type in ["OpenAI Official", "OpenAI Chat Completions", "Grok (xAI)", "OpenRouter", "DeepSeek"]:
                        assistant_first_with_tags = f"{assistant_start_tag}\n{assistant_first_message}\n\n{assistant_end_tag}\n\n{user_start_tag}"
                        accumulated_content = assistant_first_with_tags
                        full_response = assistant_first_with_tags
                    
                    # For text completions, start with the prompt ending context
                    if api_type == "OpenAI Text Completions" and prompt_ending_for_text:
                        accumulated_content = prompt_ending_for_text
                        full_response = prompt_ending_for_text

                    chunk_count = 0
                    initial_content_length = len(accumulated_content)
                    for line in response.iter_lines():
                        if stop_flag.is_set():
                            # Save whatever we have so far before stopping
                            if accumulated_content and len(accumulated_content) > initial_content_length:
                                if q:
                                    q.put(("log", "Stopped during streaming - saving partial response..."))
                                response_to_check = accumulated_content
                                try:
                                    _save_response(response_to_check, False, user_start_tag, assistant_start_tag,
                                                   assistant_end_tag, user_end_tag, output_path, start_index,
                                                   system_message, q, prompt_metadata)
                                except Exception as e:
                                    if q:
                                        q.put(("log", f"Could not save partial response: {e}"))
                            break

                        if line:
                            try:
                                line_str = line.decode('utf-8')
                                # Handle SSE format: "data: {...}" or just "{...}"
                                if line_str.startswith('data: '):
                                    line_str = line_str[6:]  # Remove "data: " prefix
                                elif line_str.strip() == '[DONE]':
                                    break
                                
                                if not line_str.strip():
                                    continue
                                
                                chunk = json.loads(line_str)
                                content = None
                                
                                # Parse based on API type
                                if api_type == "Anthropic Claude":
                                    if chunk.get('type') == 'content_block_delta':
                                        content = chunk.get('delta', {}).get('text', '')
                                elif api_type in ["OpenAI Official", "OpenAI Chat Completions", "Grok (xAI)", "OpenRouter", "DeepSeek"]:
                                    # All OpenAI-compatible chat completions use the same format
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                elif api_type == "Gemini (Google)":
                                    # Gemini uses candidates[0].content.parts[0].text structure
                                    if 'candidates' in chunk and len(chunk['candidates']) > 0:
                                        candidate = chunk['candidates'][0]
                                        if 'content' in candidate and 'parts' in candidate['content']:
                                            parts = candidate['content']['parts']
                                            if parts and len(parts) > 0:
                                                content = parts[0].get('text', '')
                                else:  # OpenAI Text Completions
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        content = chunk['choices'][0].get('text', '')
                                
                                if content:
                                    chunk_count += 1
                                    accumulated_content += content
                                    full_response += content
                                    
                                    # Log first few chunks for debugging
                                    if chunk_count <= 3 and q:
                                        q.put(("log", f"Received chunk {chunk_count}: {content[:50]}..."))

                                    # Check for error
                                    if "No Keys Available" in full_response:
                                        if q:
                                            q.put(("error", "Key Error. Stopping the program."))
                                        return

                                    # Check for end tag
                                    if api_type == "OpenAI Text Completions":
                                        if assistant_end_tag in accumulated_content:
                                            if _handle_response(accumulated_content, user_start_tag, assistant_start_tag, 
                                                                assistant_end_tag, user_end_tag, actual_min_turns, 
                                                                actual_stop_percentage, output_path, start_index, 
                                                                system_message, q, prompt_metadata):
                                                break
                                    else:
                                        if accumulated_content.endswith(assistant_end_tag):
                                            if _handle_response(full_response, user_start_tag, assistant_start_tag, 
                                                                assistant_end_tag, user_end_tag, actual_min_turns, 
                                                                actual_stop_percentage, output_path, start_index, 
                                                                system_message, q, prompt_metadata):
                                                break

                                    # Check for refusal
                                    if refusal_pattern and refusal_pattern.search(accumulated_content):
                                        if q:
                                            q.put(("log", "Refusal detected. Restarting..."))
                                        break

                                    # Check for force retry
                                    if force_retry_pattern and force_retry_pattern.search(accumulated_content):
                                        if q:
                                            q.put(("log", "Banned phrase detected. Retrying..."))
                                        break

                            except json.JSONDecodeError:
                                pass
                            except KeyError:
                                pass
                            except Exception as e:
                                if q:
                                    q.put(("log", f"Stream parsing warning: {str(e)}"))

                    # Log what we received
                    if q:
                        q.put(("log", f"Stream complete. Total chunks: {chunk_count}, Response length: {len(accumulated_content)}"))
                        if accumulated_content:
                            q.put(("log", f"Response preview (first 200 chars): {accumulated_content[:200]}"))
                    
                    # Check if we were stopped during streaming
                    was_stopped = stop_flag.is_set()
                    
                    # Handle incomplete responses
                    response_ends_properly = False
                    if api_type == "OpenAI Text Completions":
                        response_ends_properly = accumulated_content.endswith(assistant_end_tag)
                    else:
                        response_ends_properly = accumulated_content.endswith(assistant_end_tag)
                    
                    if not response_ends_properly or was_stopped:
                        response_to_check = accumulated_content
                        if q:
                            q.put(("log", f"Checking for tags - user_end_tag: {user_end_tag}, assistant_end_tag: {assistant_end_tag}"))
                            q.put(("log", f"Has user_end_tag: {user_end_tag in response_to_check}, Has assistant_end_tag: {assistant_end_tag in response_to_check}"))
                        
                        # If stopped, save partial response even if tags are incomplete
                        if was_stopped and accumulated_content and chunk_count > 0:
                            if q:
                                q.put(("log", "Saving partial response (generation was stopped)..."))
                            try:
                                _save_response(response_to_check, False, user_start_tag, assistant_start_tag,
                                               assistant_end_tag, user_end_tag, output_path, start_index,
                                               system_message, q, prompt_metadata)
                            except Exception as e:
                                if q:
                                    q.put(("log", f"Could not save partial response: {e}"))
                        elif user_end_tag in response_to_check and assistant_end_tag in response_to_check:
                            _save_response(response_to_check, False, user_start_tag, assistant_start_tag,
                                           assistant_end_tag, user_end_tag, output_path, start_index,
                                           system_message, q, prompt_metadata)
                        else:
                            if q:
                                q.put(("log", f"Missing USER or ASSISTANT tags. Response length: {len(response_to_check)}"))
                                if response_to_check:
                                    q.put(("log", f"Response content: {response_to_check[:500]}"))
                                else:
                                    q.put(("log", "No response content received from API"))
                    
                    # If we were stopped, break out of the generation loop
                    if was_stopped:
                        break

            except requests.exceptions.RequestException as e:
                if q:
                    q.put(("log", f"Request error: {e}"))
                if stop_flag.is_set():
                    break

            if stop_flag.is_set():
                if q:
                    q.put(("log", "Stopping generation..."))
                    q.put(("stopped", "Generation stopped by user"))
                return

            # Random delay
            delay = random.uniform(min_delay, max_delay)
            if q:
                q.put(("log", f"Waiting {delay:.2f} seconds before next generation..."))
            
            # Sleep with periodic checks for stop flag
            sleep_interval = 0.1
            elapsed = 0
            while elapsed < delay and not stop_flag.is_set():
                time.sleep(min(sleep_interval, delay - elapsed))
                elapsed += sleep_interval

            generation_count += 1

        if stop_flag.is_set():
            if q:
                q.put(("stopped", "Generation stopped by user"))

    except Exception as e:
        if q:
            q.put(("error", f"Worker error: {str(e)}"))
            import traceback
            q.put(("log", traceback.format_exc()))


def _handle_response(response_text, user_start_tag, assistant_start_tag, 
                    assistant_end_tag, user_end_tag, min_turns, stop_percentage,
                    output_path, start_index, system_message, q, prompt_metadata=None):
    """Handle a response and decide if we should save and stop"""
    messages = re.split(f"({user_start_tag}|{assistant_start_tag})", user_start_tag + response_text)[1:]
    gpt_turns = sum(1 for i in range(0, len(messages), 2) if messages[i] == assistant_start_tag)

    if gpt_turns > min_turns and random.random() < stop_percentage:
        if q:
            q.put(("log", "--------------------"))
            q.put(("log", "CHECKING IF CAN SAVE? YES"))
            q.put(("log", "--------------------"))
        _save_response(messages, True, user_start_tag, assistant_start_tag,
                      assistant_end_tag, user_end_tag, output_path, start_index,
                      system_message, q, prompt_metadata)
        return True
    else:
        if q:
            q.put(("log", "--------------------"))
            q.put(("log", "CHECKING IF CAN SAVE? NO"))
            q.put(("log", "--------------------"))
        return False


def _save_response(messages, preprocessed, user_start_tag, assistant_start_tag,
                   assistant_end_tag, user_end_tag, output_path, start_index, system_message, q,
                   prompt_metadata=None):
    """Save a response to a JSONL file
    
    Args:
        messages: The conversation messages
        preprocessed: Whether messages are already split
        user_start_tag: Tag marking start of user message
        assistant_start_tag: Tag marking start of assistant message
        assistant_end_tag: Tag marking end of assistant message
        user_end_tag: Tag marking end of user message
        output_path: Path to output directory
        start_index: Index to start processing messages from
        system_message: System message used
        q: Queue for logging
        prompt_metadata: Dict containing prompt configuration used for generation
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = output_path / f"{timestamp}_claude_opus_synthstruct.jsonl"

        # Split messages if not preprocessed
        if not preprocessed:
            try:
                messages = re.split(f"({user_start_tag}|{assistant_start_tag})", user_start_tag + messages)[1:]
            except Exception:
                if q:
                    q.put(("log", "Warning: Could not properly split messages, attempting partial save..."))
                messages = []

        # Build structured messages
        structured_messages = []
        if system_message:
            structured_messages.append({
                "from": "system",
                "value": system_message
            })

        # Process messages, handling incomplete ones
        try:
            for i in range(start_index, len(messages), 2):
                if i + 1 >= len(messages):
                    continue
                
                role = "human" if messages[i] == user_start_tag else "gpt"
                content = messages[i + 1].strip().replace(user_end_tag, "").replace(assistant_end_tag, "")
                if content:
                    structured_messages.append({"from": role, "value": content})
        except Exception as e:
            if q:
                q.put(("log", f"Warning: Error processing messages: {e}, saving partial data..."))

        # Only save if we have at least some content
        if not structured_messages:
            if q:
                q.put(("log", "No valid messages to save"))
            return

        # Create ShareGPT format with metadata
        sharegpt_data = {
            "id": timestamp,
            "conversations": structured_messages
        }
        
        # Add prompt metadata if provided
        if prompt_metadata:
            sharegpt_data["prompt_metadata"] = prompt_metadata

        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

        if q:
            q.put(("log", f"Response saved to {filename} (contains {len(structured_messages)} messages)"))
        else:
            print(f"\nResponse has been saved to {filename}")

    except Exception as e:
        if q:
            q.put(("log", f"Error saving response: {e}"))
            import traceback
            q.put(("log", traceback.format_exc()))
        else:
            print(f"Error saving response: {e}")
