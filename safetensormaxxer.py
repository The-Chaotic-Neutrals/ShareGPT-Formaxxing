import os
import shutil
import json
import torch
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file
from collections import defaultdict

class SafetensorMaxxer:
    def __init__(self):
        self.model_path = None
        self.output_folder = None
        self.index_filename = None
        self.discard_names = []

    def _remove_duplicate_names(self, state_dict, *, preferred_names=None, discard_names=None):
        if preferred_names is None:
            preferred_names = []
        preferred_names = set(preferred_names)
        if discard_names is None:
            discard_names = []
        discard_names = set(discard_names)

        shareds = _find_shared_tensors(state_dict)
        to_remove = defaultdict(list)
        for shared in shareds:
            complete_names = set([name for name in shared if _is_complete(state_dict[name])])
            if not complete_names:
                if len(shared) == 1:
                    name = list(shared)[0]
                    state_dict[name] = state_dict[name].clone()
                    complete_names = {name}
                else:
                    raise RuntimeError(
                        f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}. None is covering the entire storage."
                    )

            keep_name = sorted(list(complete_names))[0]

            preferred = complete_names.difference(discard_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]

            if preferred_names:
                preferred = preferred_names.intersection(complete_names)
                if preferred:
                    keep_name = sorted(list(preferred))[0]
            for name in sorted(shared):
                if name != keep_name:
                    to_remove[keep_name].append(name)
        return to_remove

    def convert_file(self, pt_filename, sf_filename, discard_names):
        loaded = torch.load(pt_filename, map_location="cpu")
        if "state_dict" in loaded:
            loaded = loaded["state_dict"]
        to_removes = self._remove_duplicate_names(loaded, discard_names=discard_names)

        metadata = {"format": "pt"}
        for kept_name, to_remove_group in to_removes.items():
            for to_remove in to_remove_group:
                if to_remove not in metadata:
                    metadata[to_remove] = kept_name
                del loaded[to_remove]
        loaded = {k: v.contiguous() for k, v in loaded.items()}

        os.makedirs(os.path.dirname(sf_filename), exist_ok=True)
        save_file(loaded, sf_filename, metadata=metadata)
        self.check_file_size(sf_filename, pt_filename)
        reloaded = load_file(sf_filename)
        for k in loaded:
            pt_tensor = loaded[k]
            sf_tensor = reloaded[k]
            if not torch.equal(pt_tensor, sf_tensor):
                raise RuntimeError(f"The output tensors do not match for key {k}")

    def check_file_size(self, sf_filename, pt_filename):
        sf_size = os.path.getsize(sf_filename)
        pt_size = os.path.getsize(pt_filename)

        if (sf_size - pt_size) / pt_size > 0.01:
            raise RuntimeError(
                f"""The file size difference is more than 1%:
             - {sf_filename}: {sf_size}
             - {pt_filename}: {pt_size}
             """
            )

    def rename(self, pt_filename):
        filename, ext = os.path.splitext(os.path.basename(pt_filename))
        local = f"{filename}.safetensors"
        local = local.replace("pytorch_model", "model")
        return local

    def convert_single_local(self, pt_filename, sf_filename, discard_names):
        try:
            self.convert_file(pt_filename, sf_filename, discard_names)
            return [sf_filename], []
        except Exception as e:
            return [], [(pt_filename, e)]

    def convert_multi_local(self, index_filename, model_path, output_folder, discard_names):
        with open(index_filename, "r") as f:
            data = json.load(f)

        filenames = set(data["weight_map"].values())
        local_filenames = []
        errors = []

        for filename in filenames:
            pt_filename = os.path.join(model_path, filename)
            sf_filename = self.rename(pt_filename)
            sf_result, sf_errors = self.convert_single_local(pt_filename, os.path.join(output_folder, sf_filename), discard_names)
            local_filenames.extend(sf_result)
            errors.extend(sf_errors)

        # Save the updated index file as well
        index_output = os.path.join(output_folder, "model.safetensors.index.json")
        with open(index_output, "w") as f:
            newdata = {k: v for k, v in data.items()}
            newmap = {k: self.rename(v) for k, v in data["weight_map"].items()}
            newdata["weight_map"] = newmap
            json.dump(newdata, f, indent=4)
        local_filenames.append(index_output)

        return local_filenames, errors

    def copy_json_files(self, src_folder, dest_folder):
        for filename in os.listdir(src_folder):
            if filename.endswith('.json'):
                src_file = os.path.join(src_folder, filename)
                dest_file = os.path.join(dest_folder, filename)
                shutil.copy(src_file, dest_file)
