"""
Multimodal worker for SynthMaxxer - handles image captioning using vision models
Supports OpenAI Vision, Anthropic Claude, Grok (xAI), and OpenRouter APIs
Outputs in HuggingFace ImageFolder format (images/ + metadata.jsonl)
Also handles Civitai image downloading
"""
import os
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFile
import io
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Any, List, Tuple
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Allow loading truncated images (some corrupted images can still be partially loaded)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional datasets import (only needed for HF dataset input, not output)
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


def encode_image_to_base64_validated(image_path):
    """Encode an image to base64 by re-encoding through PIL to ensure validity
    
    Handles truncated/corrupted images by attempting to load what's available.
    With LOAD_TRUNCATED_IMAGES enabled, PIL will try to load partial images.
    """
    try:
        # Open and process the image
        # Note: LOAD_TRUNCATED_IMAGES is enabled globally to handle corrupted images
        with Image.open(image_path) as img:
            # Load the image data (this will fail if too corrupted, even with LOAD_TRUNCATED_IMAGES)
            img.load()
            
            # Convert to RGB if necessary (removes transparency, ensures compatibility)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes buffer in JPEG format (most compatible)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # Encode to base64
            return base64.b64encode(buffer.read()).decode('utf-8')
    except (OSError, IOError, Image.UnidentifiedImageError) as e:
        error_msg = str(e).lower()
        if "truncated" in error_msg:
            raise ValueError(f"Image file is truncated (incomplete/corrupted): {os.path.basename(image_path)}. "
                           f"The file may have been interrupted during download or save. "
                           f"Try re-downloading or regenerating this image.")
        elif "cannot identify" in error_msg or "not recognized" in error_msg or isinstance(e, Image.UnidentifiedImageError):
            raise ValueError(f"Image file is corrupted or not a valid image format: {os.path.basename(image_path)}")
        else:
            raise ValueError(f"Failed to process image {os.path.basename(image_path)}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to encode image {os.path.basename(image_path)}: {e}")


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
    temp_file_path = None
    original_image_path = image_path  # Keep original for error logging
    try:
        # Check image file size before encoding (Grok API has limits)
        file_size = os.path.getsize(image_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Check image dimensions and resize if needed
        width, height = None, None
        with Image.open(image_path) as img:
            width, height = img.size
            # Grok API typically supports up to 2048x2048, but let's be conservative
            max_dimension = 2048
            if width > max_dimension or height > max_dimension:
                if q:
                    q.put(("log", f"⚠️  Image {os.path.basename(image_path)} is large ({width}x{height}), resizing to fit API limits..."))
                # Resize if too large (maintain aspect ratio)
                ratio = min(max_dimension / width, max_dimension / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                # Save resized image to temp location
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    temp_file_path = tmp_file.name
                    img_resized.save(temp_file_path, 'JPEG', quality=95)
                    image_path = temp_file_path
                    width, height = new_width, new_height  # Update dimensions for logging
        
        # Warn if file is very large (base64 increases size by ~33%)
        if file_size_mb > 15:
            if q:
                q.put(("log", f"⚠️  Warning: Image {os.path.basename(image_path)} is large ({file_size_mb:.2f} MB), may cause API errors"))
        
        # Use validated encoding to ensure image is properly formatted and decodable
        # This re-encodes through PIL to ensure the image buffer is valid
        base64_image = encode_image_to_base64_validated(image_path)
        base64_size_mb = len(base64_image) / (1024 * 1024)
        
        # Check base64 size (Grok API typically has ~20MB limit for base64 images)
        if base64_size_mb > 20:
            raise ValueError(f"Image too large after encoding ({base64_size_mb:.2f} MB). Grok API limit is ~20MB. Try resizing the image.")
        
        # Always use JPEG MIME type since we re-encode to JPEG for validation
        mime_type = 'image/jpeg'
        
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
        
        # Better error handling with detailed error messages
        if response.status_code == 404:
            error_detail = response.text
            raise ValueError(f"Grok API endpoint not found (404). Check endpoint: {endpoint}. Response: {error_detail[:200]}")
        
        if response.status_code == 400:
            # Try to parse JSON error response for better error messages
            try:
                error_json = response.json()
                error_detail = error_json.get('error', {}).get('message', response.text)
                if 'code' in error_json.get('error', {}):
                    error_code = error_json['error']['code']
                    error_detail = f"[{error_code}] {error_detail}"
            except:
                error_detail = response.text[:500]  # Show more of the error
            
            error_msg = f"HTTP 400 Bad Request: {error_detail}"
            if q:
                q.put(("log", f"❌ Error captioning {os.path.basename(original_image_path)}: {error_msg}"))
                if width and height:
                    q.put(("log", f"   Image size: {file_size_mb:.2f} MB, Base64 size: {base64_size_mb:.2f} MB, Dimensions: {width}x{height}"))
            raise ValueError(error_msg)
        
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
        # Handle other HTTP errors
        try:
            error_json = e.response.json()
            error_detail = error_json.get('error', {}).get('message', e.response.text)
        except:
            error_detail = e.response.text[:500] if e.response else str(e)
        
        error_msg = f"HTTP error: {e.response.status_code if e.response else 'Unknown'} - {error_detail}"
        if q:
            q.put(("log", f"Error captioning {os.path.basename(original_image_path)}: {error_msg}"))
        raise ValueError(error_msg)
    except Exception as e:
        if q:
            q.put(("log", f"Error captioning {os.path.basename(original_image_path)}: {str(e)}"))
        raise
    finally:
        # Clean up temp file if we created one
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


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


def ensure_output_directory(output_folder, q=None):
    """Ensure the output directory exists and return images path"""
    images_dir = os.path.join(output_folder, "images")
    os.makedirs(images_dir, exist_ok=True)
    if q:
        q.put(("log", f"📁 Output folder: {output_folder}"))
    return output_folder, images_dir


def load_processed_from_metadata(output_folder, q=None):
    """Load already processed original filenames from metadata.jsonl for resume functionality"""
    metadata_path = os.path.join(output_folder, "metadata.jsonl")
    processed = {}  # Maps original_filename -> entry data
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        original_filename = entry.get("original_filename", "")
                        if original_filename:
                            processed[original_filename] = entry
                    except json.JSONDecodeError:
                        continue
            if q and processed:
                q.put(("log", f"📂 Found {len(processed)} previously captioned images in metadata.jsonl"))
            return processed
        except Exception as e:
            if q:
                q.put(("log", f"⚠️  Warning: Could not load metadata.jsonl: {str(e)}"))
            return {}
    return {}


def save_image_caption_streaming(original_image_path, image, caption, output_folder, images_dir, current_index, q=None, prompt_metadata=None):
    """Save a single image and caption pair to the output directory in HF-ready format
    
    Images are saved with numeric filenames (00000000.png, 00000001.png, etc.)
    and metadata.jsonl is updated with the correct format for HuggingFace ImageFolder.
    
    Args:
        current_index: Current image index number for filename generation
        prompt_metadata: Optional dict containing prompt info (name, prompt text) used for this caption
    """
    try:
        # Get original filename for reference
        original_name = os.path.basename(original_image_path)
        
        # Use sequential numeric filename (HuggingFace ImageFolder format)
        numeric_filename = f"{current_index:08d}.png"
        saved_image_path = os.path.join(images_dir, numeric_filename)
        
        # Save the image
        if isinstance(image, Image.Image):
            # Ensure RGB mode for PNG
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(saved_image_path, 'PNG')
        else:
            # If it's a path, load and re-save as PNG
            with Image.open(image) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(saved_image_path, 'PNG')
        
        # Save to metadata.jsonl (single source of truth)
        save_metadata_jsonl_entry(output_folder, numeric_filename, caption, original_name, q, prompt_metadata)
        
        return saved_image_path
        
    except Exception as e:
        if q:
            q.put(("log", f"⚠️  Warning: Failed to save image/caption: {str(e)}"))
        return None


def save_metadata_jsonl_entry(output_folder, file_name, caption, original_filename, q=None, prompt_metadata=None):
    """Append a single entry to metadata.jsonl in HuggingFace ImageFolder format"""
    try:
        metadata_path = os.path.join(output_folder, "metadata.jsonl")
        entry = {
            "file_name": file_name,
            "text": caption,
            "original_filename": original_filename
        }
        
        # Add prompt metadata if provided
        if prompt_metadata:
            entry["prompt_metadata"] = prompt_metadata
        
        with open(metadata_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        if q:
            q.put(("log", f"⚠️  Warning: Failed to append to metadata.jsonl: {str(e)}"))


def image_captioning_worker(
    image_dir,
    output_folder,
    api_key,
    endpoint,
    model,
    api_type,
    caption_prompts,
    max_tokens,
    batch_size,
    max_captions,
    stop_flag,
    hf_dataset=None,
    hf_token=None,
    q=None,
):
    """Main worker function for image captioning
    
    Outputs directly to a HuggingFace ImageFolder format:
    - <output_folder>/images/00000000.png, 00000001.png, ...
    - <output_folder>/metadata.jsonl with {"file_name": "00000000.png", "text": "caption"}
    
    Args:
        caption_prompts: List of prompt dicts with 'name', 'prompt', 'temp_min', 'temp_max' keys
                        Each prompt has its own temperature range for varied generation
    """
    import random
    from collections import deque
    
    # Handle backward compatibility - convert single string prompt to list format
    if isinstance(caption_prompts, str):
        caption_prompts = [{"name": "Default", "prompt": caption_prompts, "temp_min": 0.7, "temp_max": 1.0}]
    
    # Ensure all prompts have temperature values (for backward compatibility)
    for cp in caption_prompts:
        if "temp_min" not in cp:
            cp["temp_min"] = 0.7
        if "temp_max" not in cp:
            cp["temp_max"] = 1.0
    
    if q:
        q.put(("log", f"Starting image captioning..."))
        q.put(("log", f"Output folder: {output_folder}"))
        q.put(("log", f"API type: {api_type}, Model: {model}"))
        q.put(("log", f"Caption prompts: {len(caption_prompts)} available"))
    
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
        
        # Set up output directory (creates images/ subfolder)
        output_folder, images_dir = ensure_output_directory(output_folder, q)
        
        # Load already processed images from metadata.jsonl for resume
        processed_metadata = load_processed_from_metadata(output_folder, q)
        processed_filenames = set(processed_metadata.keys())  # Set of original filenames
        
        # Check for existing captions for resume
        if processed_filenames:
            if q:
                q.put(("log", f"✓ Found {len(processed_filenames)} previously captioned images"))
                q.put(("log", f"   Will skip processed images and continue from where we left off"))
        
        # Track the next index number for new images
        current_index = len(processed_filenames)
        
        # Process images, skipping already processed ones
        processed = len(processed_filenames)  # Start count from existing
        failed = 0
        
        # Track skipped images for batched logging
        skipped_start_idx = None
        skipped_count = 0
        
        # Create a shuffled queue of prompts - pops until empty, then refills
        # This guarantees all prompts are used exactly once before any repeat
        def create_prompt_queue():
            shuffled = caption_prompts.copy()
            random.shuffle(shuffled)
            return deque(shuffled)
        
        prompt_queue = create_prompt_queue()
        prompt_queue_lock = threading.Lock()
        
        def get_next_prompt():
            """Thread-safe prompt selection from the cycling queue"""
            nonlocal prompt_queue
            with prompt_queue_lock:
                selected = prompt_queue.popleft()
                if not prompt_queue:
                    prompt_queue = create_prompt_queue()
                    if q and len(caption_prompts) > 1:
                        q.put(("log", f"♻️  Cycled through all {len(caption_prompts)} prompts, reshuffling..."))
                return selected
        
        def process_single_image(task_info: Tuple[int, str, int]) -> Tuple[int, str, bool, str, Optional[dict]]:
            """
            Process a single image - designed to run in a thread pool.
            
            Args:
                task_info: Tuple of (original_idx, image_path, assigned_index)
            
            Returns:
                Tuple of (original_idx, original_filename, success, message, prompt_meta)
            """
            original_idx, image_path, assigned_index = task_info
            original_filename = os.path.basename(image_path)
            
            try:
                # Get prompt (thread-safe)
                selected_prompt = get_next_prompt()
                prompt_name = selected_prompt.get("name", "Unknown")
                caption_prompt_text = selected_prompt.get("prompt", "")
                prompt_temp_min = selected_prompt.get("temp_min", 0.7)
                prompt_temp_max = selected_prompt.get("temp_max", 1.0)
                
                # Random temperature within range
                if prompt_temp_min == prompt_temp_max:
                    temperature = prompt_temp_min
                else:
                    temperature = round(random.uniform(prompt_temp_min, prompt_temp_max), 2)
                
                # Call the API
                caption = caption_func(
                    image_path,
                    api_key,
                    endpoint,
                    model,
                    caption_prompt_text,
                    max_tokens,
                    temperature,
                    None  # Don't pass queue to avoid log spam from threads
                )
                
                # Load image for saving
                if use_hf_images and original_idx < len(hf_images):
                    img = hf_images[original_idx].copy()
                else:
                    with Image.open(image_path) as opened_img:
                        if opened_img.mode != 'RGB':
                            opened_img = opened_img.convert('RGB')
                        img = opened_img.copy()
                
                # Build prompt metadata
                prompt_meta = {
                    "prompt_name": prompt_name,
                    "prompt_text": caption_prompt_text,
                    "model": model,
                    "api_type": api_type,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Save image and caption
                saved_path = save_image_caption_streaming(
                    image_path, img, caption, output_folder, images_dir, 
                    assigned_index, None, prompt_meta
                )
                
                if saved_path:
                    return (original_idx, original_filename, True, f"✅ {original_filename} (prompt: {prompt_name})", prompt_meta)
                else:
                    return (original_idx, original_filename, False, f"⚠️  Captioned but failed to save: {original_filename}", None)
                    
            except Exception as e:
                return (original_idx, original_filename, False, f"❌ {original_filename}: {str(e)}", None)
        
        # Filter out already processed images first
        images_to_process: List[Tuple[int, str]] = []
        for idx, image_path in enumerate(image_paths):
            original_filename = os.path.basename(image_path)
            if original_filename in processed_filenames:
                if skipped_start_idx is None:
                    skipped_start_idx = idx + 1
                skipped_count += 1
            else:
                # Log skipped range before this image
                if skipped_count > 0 and skipped_start_idx is not None:
                    if q:
                        if skipped_count == 1:
                            q.put(("log", f"⏭️  Skipped image {skipped_start_idx}/{total_images} (already processed)"))
                        else:
                            q.put(("log", f"⏭️  Skipped images {skipped_start_idx}-{skipped_start_idx + skipped_count - 1}/{total_images} ({skipped_count} images, already processed)"))
                    skipped_start_idx = None
                    skipped_count = 0
                images_to_process.append((idx, image_path))
        
        # Log any remaining skipped images
        if skipped_count > 0 and skipped_start_idx is not None:
            if q:
                if skipped_count == 1:
                    q.put(("log", f"⏭️  Skipped image {skipped_start_idx}/{total_images} (already processed)"))
                else:
                    q.put(("log", f"⏭️  Skipped images {skipped_start_idx}-{skipped_start_idx + skipped_count - 1}/{total_images} ({skipped_count} images, already processed)"))
        
        if q:
            q.put(("log", f"🚀 Processing {len(images_to_process)} images with {batch_size} parallel workers..."))
        
        # Process images in parallel batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Prepare tasks with pre-assigned indices for deterministic output ordering
            tasks = []
            for i, (original_idx, image_path) in enumerate(images_to_process):
                assigned_index = current_index + i
                tasks.append((original_idx, image_path, assigned_index))
            
            # Submit all tasks
            future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
            
            # Process results as they complete
            completed_count = 0
            for future in as_completed(future_to_task):
                if stop_flag and stop_flag.is_set():
                    if q:
                        q.put(("log", "Captioning stopped by user"))
                        q.put(("stopped", "Stopped by user"))
                    # Cancel pending futures
                    for f in future_to_task:
                        f.cancel()
                    return
                
                original_idx, original_filename, success, message, prompt_meta = future.result()
                completed_count += 1
                
                if success:
                    processed += 1
                    processed_filenames.add(original_filename)
                else:
                    failed += 1
                
                if q:
                    progress_pct = (completed_count / len(images_to_process)) * 100
                    q.put(("log", f"[{completed_count}/{len(images_to_process)} {progress_pct:.0f}%] {message}"))
        
        # Update current_index after batch
        current_index += len(images_to_process)
        
        # Check if we have any processed images
        if processed == 0:
            error_msg = "No images were successfully processed. All images failed or were skipped."
            if q:
                q.put(("error", error_msg))
                q.put(("stopped", "No images processed"))
            return
        
        # Show sample from output folder
        if processed > 0:
            try:
                # Load first entry from metadata.jsonl for preview
                metadata_path = os.path.join(output_folder, "metadata.jsonl")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            first_entry = json.loads(first_line)
                            sample_text = first_entry.get("text", "")
                            file_name = first_entry.get("file_name", "")
                            saved_image_path = os.path.join(images_dir, file_name)
                            if saved_image_path and os.path.exists(saved_image_path):
                                with Image.open(saved_image_path) as img:
                                    preview_image = img.copy()
                                if q:
                                    q.put(("log", ""))
                                    q.put(("log", "=" * 70))
                                    q.put(("log", "📸 SAMPLE RESULT:"))
                                    q.put(("log", "=" * 70))
                                    q.put(("log", f"Caption: {sample_text}"))
                                    q.put(("log", "=" * 70))
                                    q.put(("preview_image", preview_image))
            except Exception as preview_error:
                if q:
                    q.put(("log", f"⚠️  Could not load preview: {str(preview_error)}"))
        
        # Final summary
        success_msg = f"✅ Image captioning complete! Processed: {processed}, Failed: {failed}, Total: {total_images}"
        if q:
            q.put(("log", ""))
            q.put(("log", success_msg))
            q.put(("log", f"📁 Output folder: {output_folder}"))
            q.put(("log", f"   - images/ ({processed} images)"))
            q.put(("log", f"   - metadata.jsonl"))
            q.put(("log", ""))
            q.put(("log", "Ready to upload to HuggingFace! 🚀"))
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


# Civitai image downloader constants and helpers
CIVITAI_API_BASE = "https://civitai.com/api/v1/images"
CIVITAI_TAG_RE = re.compile(r"<.*?>", re.DOTALL)
CIVITAI_CONNECT_TIMEOUT = 10
CIVITAI_READ_TIMEOUT = 60
CIVITAI_TIMEOUT = (CIVITAI_CONNECT_TIMEOUT, CIVITAI_READ_TIMEOUT)
CIVITAI_API_LIMIT = 200
CIVITAI_MAX_EMPTY_BATCHES = 40
CIVITAI_API_WATCHDOG_SECONDS = 90


def make_civitai_session():
    """Create a requests session with retry logic for Civitai API"""
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)  # type: ignore[arg-type]
    session.mount("https://", adapter)
    return session


def sleep_backoff(i: int, base: float = 1.0, cap: float = 20.0):
    """Sleep with exponential backoff"""
    time.sleep(min(base * (2**i), cap))


def get_with_watchdog(session, url, *, headers, params, timeout, watchdog_seconds=90, q=None):
    """
    Runs requests.get in a background thread and enforces a hard wall-clock timeout.
    Windows-safe. If the request hangs, we bail instead of freezing forever.
    """
    result: Dict[str, Optional[Any]] = {"resp": None, "err": None}

    def _run():
        try:
            result["resp"] = session.get(url, headers=headers, params=params, timeout=timeout)
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(watchdog_seconds)

    if t.is_alive():
        error_msg = f"Hard timeout: request exceeded {watchdog_seconds}s"
        if q:
            q.put(("log", f"❌ {error_msg}"))
        raise TimeoutError(error_msg)

    if result["err"] is not None:
        raise result["err"]

    return result["resp"]


def extract_prompt_text(meta: dict) -> str:
    """Extract prompt text from Civitai image metadata"""
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
    """Normalize text for matching"""
    s = CIVITAI_TAG_RE.sub(" ", s)
    return s.lower()


def text_pass(text: str, include_terms: list[str], exclude_terms: list[str]) -> bool:
    """
    INCLUDE: OR semantics (if include_terms is non-empty, at least one must match)
    EXCLUDE: OR semantics (if any exclude term matches, reject)
    
    Uses substring matching (not exact word matching), so "slimegirl" will match "tall slimegirl"
    """
    t = normalize_text(text)
    
    # Normalize terms the same way as text for consistent matching
    normalized_include = [normalize_text(x) for x in include_terms]
    normalized_exclude = [normalize_text(x) for x in exclude_terms]

    # INCLUDE: OR logic - substring matching (any term found anywhere in text)
    if normalized_include and not any(x in t for x in normalized_include):
        return False

    # EXCLUDE: OR logic - substring matching (any term found anywhere in text)
    for x in normalized_exclude:
        if x in t:
            return False

    return True


def guess_ext(url):
    """Guess file extension from URL"""
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".jpg"


def download_bytes(session: requests.Session, url: str, q=None) -> bytes | None:
    """Download image bytes with retry logic"""
    for i in range(5):
        try:
            r = session.get(url, timeout=CIVITAI_TIMEOUT)
        except requests.RequestException as e:
            if i == 4:
                if q:
                    q.put(("log", f"❌ Failed to download {url}: {e}"))
                return None
            sleep_backoff(i)
            continue

        if r.status_code < 400:
            return r.content

        if r.status_code in (429, 500, 502, 503, 504):
            if i == 4:
                if q:
                    q.put(("log", f"❌ Failed to download {url}: HTTP {r.status_code}"))
                return None
            sleep_backoff(i)
            continue

        return None
    return None


def save_image_and_text(raw: bytes, url: str, image_id: int, out_dir: str, prompt_text: str):
    """Save image and text file to output directory"""
    ext = guess_ext(url)
    img_path = os.path.join(out_dir, f"{image_id}{ext}")
    with open(img_path, "wb") as f:
        f.write(raw)

    txt_path = os.path.join(out_dir, f"{image_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)


def load_downloaded_urls(log_path: str) -> set:
    """Load set of already downloaded URLs from log file"""
    if not os.path.exists(log_path):
        return set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return {l.strip() for l in f if l.strip()}
    except Exception:
        return set()


def mark_downloaded(log_path: str, url: str):
    """Mark URL as downloaded in log file"""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(url + "\n")
    except Exception:
        pass


def civitai_image_download_worker(
    api_key: str,
    output_dir: str,
    max_images: int,
    min_width: int,
    min_height: int,
    nsfw_level: str | None,
    include_terms: list[str],
    exclude_terms: list[str],
    sort_mode: str,
    save_meta_jsonl: bool,
    stop_flag: threading.Event,
    q=None,
    batch_size: int = 200,
    max_empty_batches: int = 40,
    wait_time: float = 0.0,
):
    """
    Worker function for downloading images from Civitai API
    
    Args:
        api_key: Civitai API key
        output_dir: Directory to save downloaded images
        max_images: Maximum number of images to download
        min_width: Minimum image width
        min_height: Minimum image height
        nsfw_level: NSFW filter level (None, "None", "Soft", "Mature", "X")
        include_terms: List of terms that must be present (OR logic)
        exclude_terms: List of terms that must not be present (OR logic)
        sort_mode: Sort mode ("Newest", "Most Reactions", "Most Comments")
        save_meta_jsonl: Whether to save metadata JSONL file
        stop_flag: Threading event to signal stop
        q: Queue for GUI communication
        batch_size: Number of images per API request (default: 200)
        max_empty_batches: Maximum number of empty batches before stopping (default: 40)
        wait_time: Wait time in seconds between page requests (default: 0.0)
    """
    if q:
        q.put(("log", "=== Civitai Image Downloader started ==="))
        q.put(("log", f"Output directory: {output_dir}"))
        q.put(("log", f"Max images: {max_images}"))
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup log file for tracking downloaded URLs
        log_path = os.path.join(output_dir, "downloaded_urls.log")
        downloaded = load_downloaded_urls(log_path)
        
        if q:
            q.put(("log", f"Found {len(downloaded)} previously downloaded images"))
        
        # Setup metadata JSONL file if requested
        meta_jsonl_path = None
        if save_meta_jsonl:
            meta_jsonl_path = os.path.join(output_dir, "image_metadata.jsonl")
            if q:
                q.put(("log", f"Metadata JSONL: {meta_jsonl_path}"))
        
        # Setup HTTP session
        session = make_civitai_session()
        headers = {"Connection": "close"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Setup API parameters
        base_params = {
            "limit": batch_size,
            "sort": sort_mode,
            "period": "AllTime",
            "withMeta": "true",
            "include": ["metaSelect", "tagIds", "profilePictures"],
        }
        
        if nsfw_level is not None:
            base_params["nsfw"] = nsfw_level
        
        has_search_terms = bool(include_terms or exclude_terms)
        # Try cursor-based pagination for all modes (more reliable than page-based)
        # The API may support cursor-based pagination for all sort modes
        use_cursor = True  # Try cursor-based for all modes
        cursor = None
        page = 1
        saved = 0
        empty_batches = 0
        max_page = 10000  # Safety limit for page-based pagination
        fallback_to_page = False  # Track if we need to fall back to page-based
        first_request = True  # Track if this is the first API request
        
        if q:
            q.put(("log", f"Starting download (batch_size={batch_size}, sort={sort_mode}, paging={'cursor' if use_cursor else 'page'})"))
        
        while saved < max_images:
            if stop_flag.is_set():
                if q:
                    q.put(("log", "Download stopped by user"))
                    q.put(("stopped", "Stopped by user"))
                break
            
            params = dict(base_params)
            
            if use_cursor and not fallback_to_page:
                if cursor:
                    params["cursor"] = cursor
                # First request - don't send cursor param, API will provide one if supported
            else:
                # Fallback to page-based pagination
                # Safety check for page-based pagination
                if page > max_page:
                    if q:
                        q.put(("log", f"⚠️  Reached maximum page limit ({max_page}), stopping"))
                    break
                params["page"] = page
            
            try:
                if q:
                    q.put(("log", f"Requesting {'cursor' if use_cursor else 'page'}={cursor if use_cursor else page}..."))
                
                r = get_with_watchdog(
                    session,
                    CIVITAI_API_BASE,
                    headers=headers,
                    params=params,
                    timeout=CIVITAI_TIMEOUT,
                    watchdog_seconds=CIVITAI_API_WATCHDOG_SECONDS,
                    q=q,
                )
                
                if r is None:
                    if q:
                        q.put(("log", "⚠️  Received None response"))
                    sleep_backoff(0)
                    continue
                
                if q:
                    q.put(("log", f"Response {r.status_code} bytes={len(r.content)}"))
                    
            except TimeoutError as e:
                if q:
                    q.put(("log", f"⚠️  {str(e)}"))
                sleep_backoff(0)
                continue
            except requests.RequestException as e:
                if q:
                    q.put(("log", f"⚠️  RequestException: {e}"))
                sleep_backoff(0)
                continue
            
            if r.status_code >= 400:
                if r.status_code in (429, 500, 502, 503, 504):
                    sleep_backoff(0)
                    continue
                error_msg = f"Request failed: {r.status_code}"
                if q:
                    q.put(("log", f"❌ {error_msg}"))
                    try:
                        error_text = r.text[:1000] if r.text else "No response text"
                        q.put(("log", f"Response: {error_text}"))
                        # Try to parse JSON error if available
                        try:
                            error_json = r.json()
                            q.put(("log", f"Error JSON: {str(error_json)[:500]}"))
                        except Exception:
                            pass
                    except Exception:
                        pass
                # For page-based pagination, if we get a 400 error, try switching to cursor if available
                if not use_cursor:
                    try:
                        error_data = r.json() if r.content else {}
                        if error_data.get("metadata", {}).get("nextCursor"):
                            if q:
                                q.put(("log", f"⚠️  Got error on page {page}, but cursor available - switching to cursor-based"))
                            use_cursor = True
                            cursor = error_data["metadata"]["nextCursor"]
                            continue
                    except Exception:
                        pass
                break
            
            data = r.json() if r.content else {}
            items = data.get("items") or []
            
            md = data.get("metadata") or {}
            
            # Check if API provides cursor in response
            available_cursor = md.get("nextCursor")
            
            # If we're trying cursor-based but API doesn't provide one on first request, fall back to page-based
            if use_cursor and not fallback_to_page and first_request:
                first_request = False
                if available_cursor is None:
                    # First request didn't provide cursor, API might not support it for this sort mode
                    if q:
                        q.put(("log", f"⚠️  API didn't provide cursor for sort mode '{sort_mode}', falling back to page-based pagination"))
                    fallback_to_page = True
                    use_cursor = False
                    page = 1  # Reset to page 1
                    continue  # Retry with page-based
                else:
                    # API provided cursor, use it
                    cursor = available_cursor
            elif use_cursor and not fallback_to_page:
                # Use the cursor from API response for subsequent requests
                cursor = available_cursor
                first_request = False
            
            if not items:
                if use_cursor:
                    cursor = md.get("nextCursor")
                    if cursor:
                        if q:
                            q.put(("log", f"Empty batch but cursor exists, continuing..."))
                        continue
                    else:
                        if q:
                            q.put(("log", "Empty batch and no cursor, stopping"))
                        break
                else:
                    # For page-based pagination, continue through empty batches like cursor-based
                    # Check if we've hit too many empty batches
                    empty_batches += 1
                    if q:
                        q.put(("log", f"Empty batch on page {page} ({empty_batches}/{max_empty_batches})"))
                    if empty_batches >= max_empty_batches:
                        if q:
                            q.put(("log", f"⚠️  Stopping: too many empty batches on page {page}"))
                        break
                    # Continue to next page
                    page += 1
                    if page > max_page:
                        if q:
                            q.put(("log", f"⚠️  Reached maximum page limit ({max_page}), stopping"))
                        break
                    continue
            
            # Only update pagination if we got items
            if use_cursor:
                cursor = md.get("nextCursor")
            else:
                page += 1
            
            matched = []
            skipped_no_url = 0
            skipped_already_downloaded = 0
            skipped_size = 0
            skipped_terms = 0
            
            for img in items:
                if stop_flag.is_set():
                    break
                
                url = img.get("url")
                if not url:
                    skipped_no_url += 1
                    continue
                
                if url in downloaded:
                    skipped_already_downloaded += 1
                    continue
                
                if img.get("width", 0) < min_width or img.get("height", 0) < min_height:
                    skipped_size += 1
                    continue
                
                prompt_text = extract_prompt_text(img.get("meta") or {})
                if has_search_terms and not text_pass(prompt_text, include_terms, exclude_terms):
                    skipped_terms += 1
                    continue
                
                matched.append((img, prompt_text))
            
            if not matched:
                empty_batches += 1
                if q:
                    q.put(("log", f"Matched 0 in this batch ({empty_batches}/{max_empty_batches}). items={len(items)}"))
                    if empty_batches == 1 or empty_batches % 10 == 0:  # Show details on first batch or every 10th
                        if skipped_no_url > 0:
                            q.put(("log", f"   - Skipped {skipped_no_url} items (no URL)"))
                        if skipped_already_downloaded > 0:
                            q.put(("log", f"   - Skipped {skipped_already_downloaded} items (already downloaded)"))
                        if skipped_size > 0:
                            q.put(("log", f"   - Skipped {skipped_size} items (size < {min_width}x{min_height})"))
                        if skipped_terms > 0:
                            q.put(("log", f"   - Skipped {skipped_terms} items (term filters)"))
                        if skipped_no_url == 0 and skipped_already_downloaded == 0 and skipped_size == 0 and skipped_terms == 0:
                            q.put(("log", f"   - All {len(items)} items passed filters but none matched (check filters)"))
                if empty_batches >= max_empty_batches:
                    if q:
                        q.put(("log", "⚠️  Stopping: too many empty batches."))
                    break
                continue
            else:
                empty_batches = 0
            
            remaining = max_images - saved
            take = min(len(matched), remaining)
            
            for img, prompt_text in matched[:take]:
                if stop_flag.is_set() or saved >= max_images:
                    break
                
                url = img["url"]
                image_id = img["id"]
                
                if q:
                    q.put(("log", f"Downloading image {saved+1}/{max_images}: {image_id}"))
                
                raw = download_bytes(session, url, q)
                if not raw:
                    continue
                
                save_image_and_text(raw, url, image_id, output_dir, prompt_text)
                
                mark_downloaded(log_path, url)
                downloaded.add(url)
                saved += 1
                
                if save_meta_jsonl and meta_jsonl_path:
                    try:
                        meta_obj = {
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
                        line = json.dumps(meta_obj, ensure_ascii=False)
                        with open(meta_jsonl_path, "a", encoding="utf-8") as f:
                            f.write(line + "\n")
                    except Exception:
                        pass
                
                if saved % 25 == 0:
                    if q:
                        q.put(("log", f"✅ Saved: {saved}/{max_images}"))
            
            if q:
                q.put(("log", f"Batch done | matched={len(matched)} | saved={saved}/{max_images}"))
            
            # Wait between page requests if configured
            if wait_time > 0 and saved < max_images and not stop_flag.is_set():
                if q:
                    q.put(("log", f"Waiting {wait_time}s before next page request..."))
                time.sleep(wait_time)
        
        success_msg = f"✅ Download complete! Downloaded {saved} images to {output_dir}"
        if q:
            q.put(("log", success_msg))
            q.put(("success", success_msg))
            q.put(("stopped", "Completed"))
            
    except Exception as e:
        error_msg = f"Civitai download failed: {str(e)}"
        if q:
            q.put(("error", error_msg))
            q.put(("stopped", "Error occurred"))
            import traceback
            q.put(("log", traceback.format_exc()))
        import traceback
        print(f"CIVITAI_DOWNLOAD_WORKER_ERROR: {error_msg}")
        print(traceback.format_exc())
