import os
import json
import random
import re

class LineMancerCore:
    def __init__(self):
        # Base directory of this script
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Repo root (parent of script directory)
        self.repo_root = os.path.dirname(self.base_dir)
        # Output folders in repo outputs directory
        outputs_dir = os.path.join(self.repo_root, "outputs")
        self.split_dir = os.path.join(outputs_dir, "linemancer", "split")
        self.merge_dir = os.path.join(outputs_dir, "linemancer", "merge")
        self.shuffle_dir = os.path.join(outputs_dir, "linemancer", "shuffled")

        # Create them if missing
        for d in [self.split_dir, self.merge_dir, self.shuffle_dir]:
            os.makedirs(d, exist_ok=True)

    def split_jsonl(self, input_path, lines_per_file):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if lines_per_file <= 0:
            raise ValueError("Lines per file must be > 0")

        prefix = os.path.splitext(os.path.basename(input_path))[0]
        index = 1
        buffer = []

        with open(input_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    buffer.append(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping invalid JSON line {i}")
                if len(buffer) >= lines_per_file:
                    self._write_split_file(self.split_dir, prefix, index, buffer)
                    index += 1
                    buffer = []
            if buffer:
                self._write_split_file(self.split_dir, prefix, index, buffer)

        print(f"[LineMancer] Split into {index} files with prefix '{prefix}' in '{self.split_dir}'")
        return index

    def _write_split_file(self, directory, prefix, index, buffer):
        out_path = os.path.join(directory, f"{prefix}_{index}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as outfile:
            outfile.write("\n".join(buffer) + "\n")
        print(f"[LineMancer] Wrote {len(buffer)} lines to {out_path}")

    def merge_jsonl(self, input_files=None, input_dir=None, prefix=None, output_filename=None):
        """
        Merge multiple JSONL files into one.
        
        Parameters:
        - input_files: list of explicit file paths to merge (if given, overrides input_dir/prefix)
        - input_dir: directory to look for files (default: self.split_dir)
        - prefix: prefix filter for files in input_dir
        - output_filename: name of output merged file (in self.merge_dir)
        
        Returns:
        - output_path, total_lines
        """
        if input_files:
            # Validate files exist
            files = []
            for f in input_files:
                if not os.path.isfile(f):
                    print(f"[Warning] Input file not found: {f}, skipping")
                else:
                    files.append(f)
            if not files:
                raise FileNotFoundError("No valid input files provided in input_files list.")
            # Sort files by extracted index if possible, else lex order
            def extract_index(filepath):
                fname = os.path.basename(filepath)
                m = re.search(r"_(\d+)\.jsonl$", fname)
                return int(m.group(1)) if m else 0
            files.sort(key=extract_index)
            input_paths = files
            # For output filename prefix extraction
            if prefix is None and len(files) > 0:
                prefix = os.path.splitext(os.path.basename(files[0]))[0].rsplit('_', 1)[0]
        else:
            input_dir = input_dir or self.split_dir

            if prefix is None and output_filename:
                prefix = os.path.splitext(output_filename)[0]

            if prefix:
                pattern = re.compile(re.escape(prefix) + r"_\d+\.jsonl$")
                files = [f for f in os.listdir(input_dir) if pattern.match(f)]
                if not files:
                    raise FileNotFoundError(f"No files with prefix '{prefix}' found in {input_dir}")
            else:
                files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
                if not files:
                    raise FileNotFoundError(f"No .jsonl files found in {input_dir}")

            def extract_index(filename):
                m = re.search(r"_(\d+)\.jsonl$", filename)
                return int(m.group(1)) if m else 0

            files.sort(key=extract_index)
            input_paths = [os.path.join(input_dir, f) for f in files]

        if output_filename is None:
            output_filename = f"{prefix}_merged.jsonl" if prefix else "merged.jsonl"
        if not output_filename.endswith(".jsonl"):
            output_filename += ".jsonl"
        output_path = os.path.join(self.merge_dir, output_filename)

        total_lines = 0
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for path in input_paths:
                if not os.path.isfile(path):
                    print(f"[Warning] File not found: {path}, skipping")
                    continue
                with open(path, 'r', encoding='utf-8') as infile:
                    for i, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                            outfile.write(line + "\n")
                            total_lines += 1
                        except json.JSONDecodeError:
                            print(f"[Warning] Skipping invalid JSON line {i} in {path}")

        print(f"[LineMancer] Merged {total_lines} lines into {output_path}")
        return output_path, total_lines

    def shuffle_jsonl(self, input_path):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        prefix = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(self.shuffle_dir, f"{prefix}_shuffled.jsonl")

        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        random.shuffle(lines)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    json.dump(obj, outfile, ensure_ascii=False)
                    outfile.write("\n")
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping invalid JSON line {i} during shuffle.")

        print(f"[LineMancer] Shuffled {len(lines)} lines into {output_path}")
        return output_path
