"""
Multimodal worker for SynthMaxxer - handles image captioning using vision models
Supports OpenAI Vision, Anthropic Claude, Grok (xAI), and OpenRouter APIs
Outputs in HuggingFace dataset format (Parquet) with text and images columns
"""
import os
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from PIL import Image
import io

try:
    from datasets import Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image {image_path}: {e}")


def get_image_mime_type(image_path):
    """Get MIME type for an image based on extension"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')


def caption_image_openai(image_path, api_key, endpoint, model, caption_prompt, max_tokens, temperature, q=None):
    """Caption an image using OpenAI Vision API"""
    try:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": caption_prompt if caption_prompt else "Describe what you see in this image in detail. Include all important elements, objects, people, text, colors, composition, and context."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        caption = result['choices'][0]['message']['content']
        return caption.strip()
        
    except Exception as e:
        if q:
            q.put(("log", f"Error captioning {os.path.basename(image_path)}: {str(e)}"))
        raise


def caption_image_claude(image_path, api_key, endpoint, model, caption_prompt, max_tokens, temperature, q=None):
    """Caption an image using Anthropic Claude API"""
    try:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        prompt_text = caption_prompt if caption_prompt else "Describe what you see in this image in detail. Include all important elements, objects, people, text, colors, composition, and context."
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        # Claude returns content in a list of content blocks
        caption = ""
        for content_block in result.get('content', []):
            if content_block.get('type') == 'text':
                caption += content_block.get('text', '')
        
        return caption.strip()
        
    except Exception as e:
        if q:
            q.put(("log", f"Error captioning {os.path.basename(image_path)}: {str(e)}"))
        raise


def caption_image_grok(image_path, api_key, endpoint, model, caption_prompt, max_tokens, temperature, q=None):
    """Caption an image using Grok (xAI) Vision API"""
    try:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt_text = caption_prompt if caption_prompt else "Describe what you see in this image in detail. Include all important elements, objects, people, text, colors, composition, and context."
        
        # Grok uses OpenAI-compatible format for vision
        # Note: Ensure endpoint is correct - should be https://api.x.ai/v1/chat/completions
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Validate endpoint
        if not endpoint or "api.x.ai" not in endpoint:
            raise ValueError(f"Invalid Grok endpoint: {endpoint}. Should be https://api.x.ai/v1/chat/completions")
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        
        # Better error handling
        if response.status_code == 404:
            error_detail = response.text
            raise ValueError(f"Grok API endpoint not found (404). Check endpoint: {endpoint}. Response: {error_detail[:200]}")
        
        response.raise_for_status()
        
        result = response.json()
        
        # Handle response format
        if 'choices' in result and len(result['choices']) > 0:
            caption = result['choices'][0]['message']['content']
        elif 'content' in result:
            caption = result['content']
        else:
            raise ValueError(f"Unexpected response format: {result}")
        
        return caption.strip()
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP error: {e.response.status_code} - {e.response.text[:200] if e.response else str(e)}"
        if q:
            q.put(("log", f"Error captioning {os.path.basename(image_path)}: {error_msg}"))
        raise ValueError(error_msg)
    except Exception as e:
        if q:
            q.put(("log", f"Error captioning {os.path.basename(image_path)}: {str(e)}"))
        raise


def caption_image_openrouter(image_path, api_key, endpoint, model, caption_prompt, max_tokens, temperature, q=None):
    """Caption an image using OpenRouter API (OpenAI-compatible)"""
    # OpenRouter uses OpenAI-compatible format
    return caption_image_openai(image_path, api_key, endpoint, model, caption_prompt, max_tokens, temperature, q)


def get_image_files(image_dir):
    """Get all supported image files from a directory"""
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def check_dataset_for_existing_captions(dataset, q=None):
    """Check if dataset already has text/caption columns"""
    if not DATASETS_AVAILABLE or not dataset:
        return None, []
    
    # Common column names for captions/text
    caption_column_names = [
        "text", "caption", "captions", "description", "descriptions",
        "label", "labels", "annotation", "annotations", "alt_text",
        "title", "titles", "summary", "summaries", "content"
    ]
    
    # Get dataset column names
    try:
        column_names = dataset.column_names if hasattr(dataset, 'column_names') else []
        
        if not column_names:
            return None, []
        
        # Check for caption-like columns
        found_columns = []
        for col_name in column_names:
            col_lower = col_name.lower()
            # Check if column name matches any caption pattern
            if any(caption_name in col_lower for caption_name in caption_column_names):
                found_columns.append(col_name)
            # Also check if column contains text data (sample first few examples)
            elif len(dataset) > 0:
                try:
                    # Try different ways to access dataset items
                    first_item = dataset[0]
                    if isinstance(first_item, dict):
                        sample_value = first_item.get(col_name)
                    elif hasattr(first_item, col_name):
                        sample_value = getattr(first_item, col_name)
                    else:
                        sample_value = None
                    
                    if isinstance(sample_value, str) and len(sample_value.strip()) > 0:
                        # Check if it looks like a caption (not just metadata)
                        if len(sample_value) > 10 and not col_lower in ['id', 'path', 'url', 'file_name']:
                            found_columns.append(col_name)
                except (KeyError, IndexError, TypeError, AttributeError):
                    pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in found_columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)
        
        # Get sample text from first found column if any
        sample_text = None
        if unique_columns:
            try:
                first_col = unique_columns[0]
                if len(dataset) > 0:
                    first_item = dataset[0]
                    if isinstance(first_item, dict):
                        sample_value = first_item.get(first_col)
                    elif hasattr(first_item, first_col):
                        sample_value = getattr(first_item, first_col)
                    else:
                        sample_value = None
                    
                    if isinstance(sample_value, str):
                        sample_text = sample_value[:200]  # First 200 chars
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
        
        return sample_text, unique_columns
        
    except Exception as e:
        if q:
            q.put(("log", f"Warning: Could not check for existing captions: {str(e)}"))
        return None, []


def load_hf_dataset(dataset_name, split=None, token=None, q=None):
    """Download and load a HuggingFace dataset (uses HF cache automatically)"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    try:
        from datasets import load_dataset
        import os
        
        # Check if dataset is cached
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        dataset_cache_path = None
        if os.path.exists(cache_dir):
            # Try to find cached dataset
            for root, dirs, files in os.walk(cache_dir):
                if dataset_name.replace("/", "___") in root or dataset_name.replace("/", "--") in root:
                    dataset_cache_path = root
                    break
        
        if q:
            if dataset_cache_path and os.path.exists(dataset_cache_path):
                q.put(("log", f"📦 Found cached dataset at: {dataset_cache_path}"))
                q.put(("log", f"   Loading from cache (no re-download needed)"))
            else:
                if token:
                    q.put(("log", f"⬇️  Downloading HuggingFace dataset: {dataset_name} (using token)"))
                else:
                    q.put(("log", f"⬇️  Downloading HuggingFace dataset: {dataset_name}"))
                q.put(("log", f"   (Will be cached for future use at ~/.cache/huggingface/datasets)"))
        
        # Prepare token parameter - use token if provided, otherwise None (will use cached token if available)
        token_param = token if token else None
        
        if split:
            dataset = load_dataset(dataset_name, split=split, token=token_param)
        else:
            dataset_dict = load_dataset(dataset_name, token=token_param)
            # Use the first split if multiple splits available
            if isinstance(dataset_dict, DatasetDict):
                # Type narrowing: dataset_dict is confirmed to be DatasetDict here
                dataset_dict_typed: DatasetDict = dataset_dict
                split_name = list(dataset_dict_typed.keys())[0]
                dataset = dataset_dict_typed[split_name]
                if q:
                    q.put(("log", f"Using split: {split_name}"))
            else:
                dataset = dataset_dict
        
        if q:
            # Check if dataset has length (IterableDataset doesn't support len())
            try:
                from datasets import IterableDataset
                if isinstance(dataset, IterableDataset):
                    q.put(("log", f"✅ Loaded dataset (IterableDataset - size unknown)"))
                else:
                    q.put(("log", f"✅ Loaded dataset with {len(dataset)} examples"))
            except (TypeError, AttributeError):
                # Fallback if len() fails for any reason
                q.put(("log", f"✅ Loaded dataset (size unknown)"))
        
        # Check for existing captions/text
        sample_text, caption_columns = check_dataset_for_existing_captions(dataset, q)
        if caption_columns:
            if q:
                q.put(("log", f"⚠️  Dataset already has text/caption columns: {', '.join(caption_columns)}"))
                if sample_text:
                    q.put(("log", f"   Sample text: {sample_text}..."))
                q.put(("log", f"   Will generate new captions and add to 'text' column (existing columns preserved)"))
        else:
            if q:
                q.put(("log", f"✓ No existing captions found - will generate new captions"))
        
        return dataset
    except Exception as e:
        if q:
            q.put(("log", f"Error loading HuggingFace dataset: {str(e)}"))
        raise


def extract_images_from_hf_dataset(dataset, image_column="image", q=None):
    """Extract images from a HuggingFace dataset and return as list of PIL Images"""
    import tempfile
    
    temp_dir = tempfile.mkdtemp(prefix="synthmaxxer_hf_images_")
    
    if q:
        q.put(("log", f"Extracting images from dataset to temporary directory..."))
    
    images = []
    image_paths = []
    
    # Try to get images using dataset's built-in image loading if available
    # This handles relative paths and various formats automatically
    try:
        # Check if dataset has image feature that auto-loads
        if hasattr(dataset, 'features') and image_column in dataset.features:
            feature = dataset.features[image_column]
            # If it's an Image feature, it should auto-load
            if hasattr(feature, 'decode_example'):
                if q:
                    q.put(("log", f"   Using dataset's image feature for automatic loading"))
    except Exception:
        pass
    
    for idx, example in enumerate(dataset):
        try:
            if image_column not in example:
                continue
                
            image = example[image_column]
            img = None
            image_path = None
            
            # Handle different image formats - prioritize PIL Image objects
            if isinstance(image, Image.Image):
                # Already a PIL Image - use directly (best case)
                img = image
            elif isinstance(image, dict):
                # Image dict - try multiple ways to get the image
                # Some datasets store Image objects in dicts
                if 'image' in image and isinstance(image['image'], Image.Image):
                    img = image['image']
                elif 'bytes' in image:
                    img = Image.open(io.BytesIO(image['bytes']))
                elif 'path' in image:
                    path = image['path']
                    # Only try absolute paths that exist
                    if os.path.isabs(path) and os.path.exists(path):
                        img = Image.open(path)
                    # For relative paths, try to access the actual Image from dataset
                    # by using the dataset's decode functionality
                    elif hasattr(dataset, 'features') and image_column in dataset.features:
                        try:
                            # Try to get the decoded image from the feature
                            feature = dataset.features[image_column]
                            if hasattr(feature, 'decode_example'):
                                img = feature.decode_example(image)
                        except Exception:
                            pass
            elif isinstance(image, str):
                # String path - only use if absolute and exists
                if os.path.isabs(image) and os.path.exists(image):
                    img = Image.open(image)
                else:
                    # Relative path - try to use dataset's decode if available
                    if hasattr(dataset, 'features') and image_column in dataset.features:
                        try:
                            feature = dataset.features[image_column]
                            if hasattr(feature, 'decode_example'):
                                img = feature.decode_example(image)
                        except Exception:
                            pass
            elif hasattr(image, 'read'):
                # File-like object
                img = Image.open(image)
            elif hasattr(image, 'image') and isinstance(image.image, Image.Image):
                # Wrapped Image object
                img = image.image
            
            # If we still don't have an image, try accessing the dataset item directly
            # which might trigger automatic image loading
            if img is None:
                try:
                    # Try accessing the column again - sometimes HF datasets lazy-load
                    direct_image = dataset[idx][image_column]
                    if isinstance(direct_image, Image.Image):
                        img = direct_image
                except Exception:
                    pass
            
            # Convert to RGB if we have an image
            if img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Always save to temp directory for captioning API (ensures valid file path)
                image_path = os.path.join(temp_dir, f"image_{idx:06d}.png")
                img.save(image_path)
                # Create a copy to avoid keeping file handles open
                img_copy = img.copy()
                images.append(img_copy)
                image_paths.append(image_path)
                # Close the original image if it has a file handle
                if hasattr(img, 'fp') and img.fp:
                    try:
                        img.close()
                    except Exception:
                        pass
            else:
                # Log first few failures for debugging
                if q and idx < 10:
                    q.put(("log", f"⚠️  Could not extract image at index {idx}, type: {type(image)}"))
                    
        except Exception as e:
            if q and idx < 10:  # Only log first few errors to avoid spam
                q.put(("log", f"Error extracting image {idx}: {str(e)}"))
            continue
    
    if q:
        q.put(("log", f"✓ Extracted {len(images)}/{len(dataset)} images to temporary directory"))
        if len(images) == 0:
            q.put(("log", f"⚠️  No images could be extracted - check image column format"))
            q.put(("log", f"   Try checking dataset.features to see available columns"))
    
    return images, image_paths, temp_dir


def get_work_directory(output_file):
    """Get the working directory path for streaming saves"""
    # Create a .work directory next to the output file
    output_path = Path(output_file)
    work_dir = output_path.parent / f"{output_path.stem}.work"
    return str(work_dir)


def ensure_work_directory(output_file, q=None):
    """Ensure the working directory exists and return its path"""
    work_dir = get_work_directory(output_file)
    images_dir = os.path.join(work_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    if q:
        q.put(("log", f"📁 Working directory: {work_dir}"))
    return work_dir, images_dir


def load_caption_index(work_dir, q=None):
    """Load the JSON index file mapping image paths to captions"""
    index_file = os.path.join(work_dir, "captions_index.json")
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            if q:
                q.put(("log", f"📂 Loaded caption index with {len(index)} entries"))
            return index
        except Exception as e:
            if q:
                q.put(("log", f"⚠️  Warning: Could not load caption index: {str(e)}"))
            return {}
    return {}


def save_caption_index(work_dir, index, q=None):
    """Save the JSON index file mapping image paths to captions"""
    index_file = os.path.join(work_dir, "captions_index.json")
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        if q:
            q.put(("log", f"⚠️  Warning: Failed to save caption index: {str(e)}"))


def save_image_caption_streaming(original_image_path, image, caption, images_dir, index, q=None):
    """Save a single image and caption pair to the working directory"""
    try:
        # Get a safe filename from the original path
        original_name = os.path.basename(original_image_path)
        name_without_ext = os.path.splitext(original_name)[0]
        # Use index to ensure unique filenames
        idx = len(index)
        safe_filename = f"{idx:06d}_{name_without_ext}.png"
        saved_image_path = os.path.join(images_dir, safe_filename)
        
        # Save the image
        if isinstance(image, Image.Image):
            image.save(saved_image_path)
        else:
            # If it's a path, copy it
            import shutil
            shutil.copy2(image, saved_image_path)
        
        # Update index
        index[original_image_path] = {
            "caption": caption,
            "saved_image_path": saved_image_path,
            "index": idx,
            "original_filename": original_name
        }
        
        # Save index immediately
        work_dir = os.path.dirname(images_dir)
        save_caption_index(work_dir, index, q)
        
        return saved_image_path
        
    except Exception as e:
        if q:
            q.put(("log", f"⚠️  Warning: Failed to save image/caption: {str(e)}"))
        return None


def pack_work_directory_to_parquet(work_dir, output_file, q=None):
    """Pack the working directory contents into a Parquet file"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    try:
        # Load the index
        index = load_caption_index(work_dir, q)
        if not index:
            if q:
                q.put(("log", "⚠️  No captions found in working directory"))
            return False
        
        # Sort by index to maintain order
        sorted_items = sorted(index.items(), key=lambda x: x[1].get("index", 0))
        
        texts = []
        images = []
        
        images_dir = os.path.join(work_dir, "images")
        
        if q:
            q.put(("log", f"📦 Packing {len(sorted_items)} items to Parquet..."))
        
        for original_path, entry in sorted_items:
            caption = entry.get("caption", "")
            saved_image_path = entry.get("saved_image_path", "")
            
            if not caption or not saved_image_path:
                continue
            
            if not os.path.exists(saved_image_path):
                if q:
                    q.put(("log", f"⚠️  Warning: Image not found: {saved_image_path}"))
                continue
            
            try:
                # Load image
                with Image.open(saved_image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_copy = img.copy()
                images.append(img_copy)
                texts.append(caption)
            except Exception as e:
                if q:
                    q.put(("log", f"⚠️  Warning: Could not load image {saved_image_path}: {str(e)}"))
                continue
        
        if len(texts) == 0 or len(images) == 0:
            if q:
                q.put(("log", "⚠️  No valid data to pack"))
            return False
        
        # Ensure arrays match
        min_length = min(len(texts), len(images))
        if len(texts) != len(images):
            if q:
                q.put(("log", f"⚠️  Warning: Mismatch between texts ({len(texts)}) and images ({len(images)}), using {min_length} items"))
            texts = texts[:min_length]
            images = images[:min_length]
        
        # Create dataset
        dataset_dict = {
            "text": texts,
            "image": images
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Ensure output file has .parquet extension
        if not output_file.endswith('.parquet'):
            output_file = output_file.replace('.jsonl', '.parquet')
            if not output_file.endswith('.parquet'):
                output_file += '.parquet'
        
        # Save as Parquet
        dataset.to_parquet(output_file)
        
        if q:
            q.put(("log", f"✅ Packed {len(texts)} examples to {output_file}"))
        
        return True
        
    except Exception as e:
        if q:
            q.put(("log", f"❌ Error packing to Parquet: {str(e)}"))
        raise


def image_captioning_worker(
    image_dir,
    output_file,
    api_key,
    endpoint,
    model,
    api_type,
    caption_prompt,
    max_tokens,
    temperature,
    batch_size,
    max_captions,
    stop_flag,
    hf_dataset=None,
    hf_token=None,
    q=None,
):
    """Main worker function for image captioning"""
    if q:
        q.put(("log", f"Starting image captioning..."))
        q.put(("log", f"Output file: {output_file}"))
        q.put(("log", f"API type: {api_type}, Model: {model}"))
    
    temp_dir_to_cleanup = None
    
    try:
        # Handle HuggingFace dataset input
        if hf_dataset:
            if q:
                if hf_token:
                    q.put(("log", f"Loading HuggingFace dataset: {hf_dataset} (with token)"))
                else:
                    q.put(("log", f"Loading HuggingFace dataset: {hf_dataset}"))
            dataset = load_hf_dataset(hf_dataset, token=hf_token, q=q)
            hf_images, image_paths, temp_dir_to_cleanup = extract_images_from_hf_dataset(dataset, q=q)
            if not image_paths:
                error_msg = "No images found in HuggingFace dataset"
                if q:
                    q.put(("error", error_msg))
                return
            # Use HF images directly for dataset creation
            use_hf_images = True
        else:
            use_hf_images = False
            # Get all image files from directory
            if not image_dir or not os.path.isdir(image_dir):
                error_msg = f"Invalid image directory: {image_dir}"
                if q:
                    q.put(("error", error_msg))
                return
            
            image_paths = get_image_files(image_dir)
        
        total_images = len(image_paths)
        
        if total_images == 0:
            error_msg = f"No supported image files found"
            if q:
                q.put(("error", error_msg))
            return
        
        # Apply max_captions limit if set (0 = unlimited)
        if max_captions > 0 and max_captions < total_images:
            image_paths = image_paths[:max_captions]
            if q:
                q.put(("log", f"Found {total_images} image(s), limiting to {max_captions} as specified"))
        else:
            if q:
                q.put(("log", f"Found {total_images} image(s) to process"))
        
        total_images = len(image_paths)
        
        # Select captioning function based on API type
        if api_type == "OpenAI Vision":
            caption_func = caption_image_openai
        elif api_type == "Anthropic Claude":
            caption_func = caption_image_claude
        elif api_type == "Grok (xAI)":
            caption_func = caption_image_grok
        elif api_type == "OpenRouter":
            caption_func = caption_image_openrouter
        else:
            error_msg = f"Unsupported API type: {api_type}"
            if q:
                q.put(("error", error_msg))
            return
        
        if not DATASETS_AVAILABLE:
            error_msg = "datasets library is required for Parquet output. Install with: pip install datasets"
            if q:
                q.put(("error", error_msg))
            return
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up working directory for streaming saves
        work_dir, images_dir = ensure_work_directory(output_file, q)
        caption_index = load_caption_index(work_dir, q)
        
        # Check for existing captions in working directory for resume
        processed_paths = set()
        if caption_index:
            processed_paths = set(caption_index.keys())
            if q:
                q.put(("log", f"✓ Found {len(processed_paths)} previously captioned images in working directory"))
                q.put(("log", f"   Will skip processed images and continue from where we left off"))
        
        # Process images, skipping already processed ones
        processed = len(processed_paths)  # Start count from existing
        failed = 0
        
        # Process images, skipping already processed ones
        for idx, image_path in enumerate(image_paths):
            if stop_flag and stop_flag.is_set():
                if q:
                    q.put(("log", "Captioning stopped by user"))
                    q.put(("stopped", "Stopped by user"))
                return
            
            # Skip if already processed (check by path, not index)
            if image_path in processed_paths:
                if q:
                    q.put(("log", f"⏭️  Skipping image {idx+1}/{total_images} (already processed): {os.path.basename(image_path)}"))
                continue
            
            try:
                if q:
                    q.put(("log", f"Processing image {idx+1}/{total_images}: {os.path.basename(image_path)}"))
                
                caption = caption_func(
                    image_path,
                    api_key,
                    endpoint,
                    model,
                    caption_prompt,
                    max_tokens,
                    temperature,
                    q
                )
                
                # Get image for saving
                if use_hf_images and idx < len(hf_images):
                    # Use pre-extracted HF image - make a copy to avoid file handle issues
                    img = hf_images[idx].copy()
                else:
                    # Load image from path
                    try:
                        with Image.open(image_path) as opened_img:
                            # Convert to RGB if necessary
                            if opened_img.mode != 'RGB':
                                opened_img = opened_img.convert('RGB')
                            # Create a copy to close the file handle (copy doesn't keep file handle open)
                            img = opened_img.copy()
                    except Exception as img_e:
                        if q:
                            q.put(("log", f"Warning: Could not load image {image_path}: {str(img_e)}"))
                        failed += 1
                        continue
                
                # Save image and caption immediately (streaming save)
                saved_path = save_image_caption_streaming(image_path, img, caption, images_dir, caption_index, q)
                if saved_path:
                    processed += 1
                    processed_paths.add(image_path)
                    if q:
                        q.put(("log", f"✅ Captioned and saved image {idx+1}/{total_images}: {os.path.basename(image_path)}"))
                else:
                    failed += 1
                    if q:
                        q.put(("log", f"⚠️  Captioned but failed to save image {idx+1}/{total_images}: {os.path.basename(image_path)}"))
            
            except Exception as e:
                failed += 1
                if q:
                    q.put(("log", f"❌ Failed to caption {os.path.basename(image_path)}: {str(e)}"))
                continue
        
        # Check if we have any processed images
        if processed == 0:
            error_msg = "No images were successfully processed. All images failed or were skipped."
            if q:
                q.put(("error", error_msg))
                q.put(("stopped", "No images processed"))
            return
        
        # Pack working directory to Parquet
        if q:
            q.put(("log", f"📦 Packing {processed} processed images to Parquet format..."))
        
        try:
            pack_work_directory_to_parquet(work_dir, output_file, q)
        except Exception as e:
            error_msg = f"Failed to pack dataset to Parquet: {str(e)}"
            if q:
                q.put(("error", error_msg))
                q.put(("log", f"⚠️  Working directory still contains all data: {work_dir}"))
            raise
        
        # Load saved file and display sample
        if processed > 0:
            try:
                if q:
                    q.put(("log", ""))
                    q.put(("log", "📖 Loading saved file to verify..."))
                
                # Ensure output file has .parquet extension
                preview_file = output_file
                if not preview_file.endswith('.parquet'):
                    preview_file = preview_file.replace('.jsonl', '.parquet')
                    if not preview_file.endswith('.parquet'):
                        preview_file += '.parquet'
                
                # Load the saved parquet file to verify it was saved correctly
                saved_dataset = Dataset.from_parquet(preview_file)
                
                # Check if dataset has length (IterableDataset doesn't support len())
                # Also handle DatasetDict by getting the first split
                sample_text = None
                sample_image = None
                
                try:
                    # Handle DatasetDict by getting the first split
                    if hasattr(saved_dataset, 'keys') and callable(getattr(saved_dataset, 'keys', None)):
                        # It's a DatasetDict
                        first_key = list(saved_dataset.keys())[0]  # type: ignore
                        saved_dataset = saved_dataset[first_key]  # type: ignore
                    
                    # Check if it's an IterableDataset (doesn't support len() or indexing)
                    from datasets import IterableDataset
                    is_iterable = isinstance(saved_dataset, IterableDataset)
                    
                    if is_iterable:
                        # IterableDataset - use iterator
                        first_example = next(iter(saved_dataset))
                        if isinstance(first_example, dict):
                            sample_text = first_example.get("text", "No caption available")
                            sample_image = first_example.get("image", None)
                        else:
                            sample_text = "No caption available"
                            sample_image = None
                    else:
                        # Regular Dataset - can use len() and indexing
                        dataset_len = len(saved_dataset)  # type: ignore
                        if dataset_len > 0:
                            first_example = saved_dataset[0]  # type: ignore
                            if isinstance(first_example, dict):
                                sample_text = first_example.get("text", "No caption available")
                                sample_image = first_example.get("image", None)
                            else:
                                sample_text = "No caption available"
                                sample_image = None
                except Exception as dataset_error:
                    if q:
                        q.put(("log", f"⚠️  Error accessing saved dataset: {str(dataset_error)}"))
                    sample_text = None
                    sample_image = None
                
                if sample_text and sample_image:
                    
                    # Handle different image formats from saved dataset
                    if sample_image:
                        # If it's already a PIL Image, use it directly (make a copy to be safe)
                        if isinstance(sample_image, Image.Image):
                            preview_image = sample_image.copy()
                        # If it's a dict with path or bytes, try to load it
                        elif isinstance(sample_image, dict):
                            if 'path' in sample_image and os.path.exists(sample_image['path']):
                                with Image.open(sample_image['path']) as img:
                                    preview_image = img.copy()
                            elif 'bytes' in sample_image:
                                preview_image = Image.open(io.BytesIO(sample_image['bytes']))
                                # BytesIO doesn't need explicit closing, but make a copy to be safe
                                preview_image = preview_image.copy()
                            else:
                                preview_image = None
                        # If it's a string path, try to open it
                        elif isinstance(sample_image, str) and os.path.exists(sample_image):
                            with Image.open(sample_image) as img:
                                preview_image = img.copy()
                        else:
                            preview_image = None
                    else:
                        preview_image = None
                    
                    if q:
                        q.put(("log", ""))
                        q.put(("log", "=" * 70))
                        q.put(("log", "📸 SAMPLE RESULT (from saved file):"))
                        q.put(("log", "=" * 70))
                        q.put(("log", f"Caption: {sample_text}"))
                        q.put(("log", "=" * 70))
                        q.put(("log", ""))
                    
                    # Send image for preview
                    if preview_image:
                        if q:
                            q.put(("preview_image", preview_image))
                    elif q:
                        q.put(("log", "⚠️  Could not extract image for preview"))
                else:
                    if q:
                        if sample_text is None and sample_image is None:
                            q.put(("log", "⚠️  Saved file is empty or could not be read"))
                        else:
                            q.put(("log", "⚠️  Could not extract image for preview"))
                        
            except Exception as preview_error:
                if q:
                    q.put(("log", f"⚠️  Could not load preview from saved file: {str(preview_error)}"))
                    # Fallback: try loading from working directory
                    try:
                        index = load_caption_index(work_dir, q)
                        if index:
                            # Get first entry
                            sorted_items = sorted(index.items(), key=lambda x: x[1].get("index", 0))
                            if sorted_items:
                                first_path, first_entry = sorted_items[0]
                                sample_text = first_entry.get("caption", "")
                                saved_image_path = first_entry.get("saved_image_path", "")
                                if saved_image_path and os.path.exists(saved_image_path):
                                    with Image.open(saved_image_path) as img:
                                        preview_image = img.copy()
                                    q.put(("log", ""))
                                    q.put(("log", "=" * 70))
                                    q.put(("log", "📸 SAMPLE RESULT (from working directory):"))
                                    q.put(("log", "=" * 70))
                                    q.put(("log", f"Caption: {sample_text}"))
                                    q.put(("log", "=" * 70))
                                    q.put(("preview_image", preview_image))
                    except Exception as fallback_error:
                        if q:
                            q.put(("log", f"⚠️  Could not load preview from working directory: {str(fallback_error)}"))
        
        # Final summary
        success_msg = f"Image captioning complete! Processed: {processed}, Failed: {failed}, Total: {total_images}. Saved to {output_file}"
        if q:
            q.put(("log", success_msg))
            q.put(("success", success_msg))
            q.put(("stopped", "Completed"))
        
    except Exception as e:
        error_msg = f"Image captioning failed: {str(e)}"
        if q:
            q.put(("error", error_msg))
            q.put(("stopped", "Error occurred"))
        import traceback
        print(f"IMAGE_CAPTIONING_WORKER_ERROR: {error_msg}")
        print(traceback.format_exc())
    finally:
        # Cleanup temporary directory if created (only after processing is complete)
        if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
            try:
                import shutil
                # Small delay to ensure all file handles are closed
                import time
                time.sleep(0.5)
                shutil.rmtree(temp_dir_to_cleanup)
                if q:
                    q.put(("log", f"🧹 Cleaned up temporary directory"))
            except Exception as e:
                if q:
                    q.put(("log", f"Warning: Could not cleanup temp directory: {str(e)}"))
