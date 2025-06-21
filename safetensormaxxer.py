import os
import json
import time
import shutil
import torch
from collections import defaultdict
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file, safe_open
from transformers import AutoConfig, AutoTokenizer

INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"

class SafetensorMaxxer:
    def __init__(self):
        self.discard_names = []
        self.model_path = ""
        self.index_filename = ""
        self.output_folder = ""

    def _remove_duplicate_names(self, state_dict, *, preferred_names=None, discard_names=None):
        preferred_names = set(preferred_names or [])
        discard_names = set(discard_names or [])
        shareds = _find_shared_tensors(state_dict)
        to_remove = defaultdict(list)

        for shared in shareds:
            complete_names = {n for n in shared if _is_complete(state_dict[n])}
            if not complete_names:
                if len(shared) == 1:
                    name = list(shared)[0]
                    state_dict[name] = state_dict[name].clone()
                    complete_names = {name}
                else:
                    raise RuntimeError(f"No complete tensor among: {shared}")

            keep_name = sorted(complete_names)[0]
            if preferred := complete_names - discard_names:
                keep_name = sorted(preferred)[0]
            if preferred_names:
                preferred = preferred_names & complete_names
                if preferred:
                    keep_name = sorted(preferred)[0]

            for name in shared:
                if name != keep_name:
                    to_remove[keep_name].append(name)

        return to_remove

    def convert_file(self, pt_filename, sf_filename):
        loaded = torch.load(pt_filename, map_location="cpu")
        if "state_dict" in loaded:
            loaded = loaded["state_dict"]

        to_removes = self._remove_duplicate_names(loaded, discard_names=self.discard_names)
        metadata = {"format": "pt"}

        for kept_name, to_remove_group in to_removes.items():
            for to_remove in to_remove_group:
                metadata[to_remove] = kept_name
                del loaded[to_remove]

        loaded = {k: v.contiguous() for k, v in loaded.items()}
        os.makedirs(os.path.dirname(sf_filename), exist_ok=True)
        save_file(loaded, sf_filename, metadata=metadata)
        self._verify_conversion(sf_filename, pt_filename, loaded)

    def _verify_conversion(self, sf_filename, pt_filename, loaded):
        if abs(os.path.getsize(sf_filename) - os.path.getsize(pt_filename)) / os.path.getsize(pt_filename) > 0.01:
            raise RuntimeError("File size difference >1% after conversion.")

        reloaded = load_file(sf_filename)
        for k in loaded:
            if not torch.equal(loaded[k], reloaded[k]):
                raise RuntimeError(f"Tensor mismatch after conversion for key: {k}")

    def convert_sharded(self, index_filename, model_path, output_folder):
        with open(index_filename) as f:
            data = json.load(f)

        filenames = set(data["weight_map"].values())
        new_map = {}

        for fname in filenames:
            pt_file = os.path.join(model_path, fname)
            new_name = fname.replace("pytorch_model", "model").replace(".bin", ".safetensors")
            out_file = os.path.join(output_folder, new_name)
            self.convert_file(pt_file, out_file)
            for k, v in data["weight_map"].items():
                if v == fname:
                    new_map[k] = new_name

        new_index = os.path.join(output_folder, "model.safetensors.index.json")
        with open(new_index, "w") as f:
            json.dump({"weight_map": new_map}, f, indent=4)

        return new_index

    def scan_safetensors_files(self, folder):
        tensor_map = {}
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".safetensors"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, folder)
                    with safe_open(full_path, framework="pt") as f:
                        for key in f.keys():
                            if key in tensor_map:
                                raise ValueError(f"Duplicate key '{key}' found in multiple files!")
                            tensor_map[key] = rel_path
        return tensor_map

    def verify_and_fix_index(self):
        index_file = os.path.join(self.output_folder, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            return ["Index file not found."]

        with open(index_file, "r", encoding="utf-8") as f:
            index_map = json.load(f)["weight_map"]

        tensor_map = self.scan_safetensors_files(self.output_folder)

        config_path = os.path.join(self.output_folder, "config.json")
        if not os.path.exists(config_path):
            return ["Missing config.json"]

        config = AutoConfig.from_pretrained(self.output_folder)
        errors = []

        for key, fname in index_map.items():
            full_path = os.path.join(self.output_folder, fname)
            if not os.path.exists(full_path):
                errors.append(f"🧨 Missing file: {full_path}")
                continue
            with safe_open(full_path, framework="pt") as f:
                if key not in f.keys():
                    errors.append(f"❌ Key '{key}' missing in {full_path}")
                else:
                    t = f.get_tensor(key)
                    if "attention" in key and "query" in key:
                        if t.shape != (config.hidden_size, config.hidden_size):
                            errors.append(f"⚠️ Shape mismatch: {key} -> {t.shape}")

        for key, src in tensor_map.items():
            if key not in index_map:
                errors.append(f"🔁 Tensor '{key}' in {src} not listed in index!")

        if errors:
            self.fix_index(tensor_map)
            errors += getattr(self, "fix_log", [])

        return errors

    def fix_index(self, tensor_map):
        index_file = os.path.join(self.output_folder, "model.safetensors.index.json")
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = index_file.replace(".json", f".backup.{ts}.json")
        shutil.copy(index_file, backup)
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump({"weight_map": tensor_map}, f, indent=4)

        self.fix_log = [
            f"🛠️ Index rewritten with {len(tensor_map)} tensor keys.",
            f"📦 Backup saved to: {os.path.basename(backup)}"
        ]

    def show_token_info(self):
        try:
            config = AutoConfig.from_pretrained(self.output_folder)
            tokenizer = AutoTokenizer.from_pretrained(self.output_folder)
            print("\n🧵 Token info:")
            for k in ["bos_token", "eos_token", "pad_token", "unk_token"]:
                v = getattr(tokenizer, k, None)
                if v:
                    print(f"  {k}: {repr(v)}")
            tmpl = getattr(config, "chat_template", None) or getattr(tokenizer, "chat_template", None)
            if tmpl:
                print("\n📨 Chat template preview:")
                for line in tmpl.strip().splitlines()[:5]:
                    print(f"    {line}")
                if len(tmpl.splitlines()) > 5:
                    print("    ...")
            else:
                print("\n📨 No chat template found.")
        except Exception as e:
            print(f"⚠️ Failed to show token info: {e}")

    def show_chat_preview(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.output_folder)
            if not hasattr(tokenizer, "apply_chat_template"):
                print("🤷 Template rendering not supported.")
                return
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the capital of France?"}
            ]
            out = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            print("\n🧪 Rendered prompt:")
            for line in out.strip().splitlines()[:10]:
                print(f"    {line}")
            if len(out.strip().splitlines()) > 10:
                print("    ...")
        except Exception as e:
            print(f"⚠️ Failed to render chat template: {e}")
