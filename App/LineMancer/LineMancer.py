import os
import json
import random
import re


class LineMancerCore:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.repo_root = os.path.dirname(os.path.dirname(self.base_dir))
        self.outputs_dir = os.path.join(self.repo_root, "Outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

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
                    self._write_split_file(self.outputs_dir, prefix, index, buffer)
                    index += 1
                    buffer = []
            if buffer:
                self._write_split_file(self.outputs_dir, prefix, index, buffer)

        print(f"[LineMancer] Split into {index} files with prefix '{prefix}' in '{self.outputs_dir}'")
        return index

    def _write_split_file(self, directory, prefix, index, buffer):
        out_path = os.path.join(directory, f"{prefix}_split_{index}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as outfile:
            outfile.write("\n".join(buffer) + "\n")
        print(f"[LineMancer] Wrote {len(buffer)} lines to {out_path}")

    def merge_jsonl(self, input_files=None, input_dir=None, prefix=None, output_filename=None):
        if input_files:
            files = []
            for f in input_files:
                if not os.path.isfile(f):
                    print(f"[Warning] Input file not found: {f}, skipping")
                else:
                    files.append(f)
            if not files:
                raise FileNotFoundError("No valid input files provided in input_files list.")
            def extract_index(filepath):
                fname = os.path.basename(filepath)
                m = re.search(r"_split_(\d+)\.jsonl$", fname)
                return int(m.group(1)) if m else 0
            files.sort(key=extract_index)
            input_paths = files
            if prefix is None and len(files) > 0:
                prefix = os.path.splitext(os.path.basename(files[0]))[0].rsplit('_', 1)[0]
        else:
            input_dir = input_dir or self.outputs_dir

            if prefix is None and output_filename:
                prefix = os.path.splitext(output_filename)[0]

            if prefix:
                pattern = re.compile(re.escape(prefix) + r"_split_\d+\.jsonl$")
                files = [f for f in os.listdir(input_dir) if pattern.match(f)]
                if not files:
                    raise FileNotFoundError(f"No split files with prefix '{prefix}' found in {input_dir}")
            else:
                files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
                if not files:
                    raise FileNotFoundError(f"No .jsonl files found in {input_dir}")

            def extract_index(filename):
                m = re.search(r"_split_(\d+)\.jsonl$", filename)
                return int(m.group(1)) if m else 0

            files.sort(key=extract_index)
            input_paths = [os.path.join(input_dir, f) for f in files]

        if output_filename is None:
            output_filename = f"{prefix}_merged.jsonl" if prefix else "merged.jsonl"
        if not output_filename.endswith(".jsonl"):
            output_filename += ".jsonl"
        output_path = os.path.join(self.outputs_dir, output_filename)

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
        output_path = os.path.join(self.outputs_dir, f"{prefix}_shuffled.jsonl")

        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = [line for line in infile if line.strip()]

        random.shuffle(lines)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in lines:
                outfile.write(line)

        print(f"[LineMancer] Shuffled {len(lines)} lines into {output_path}")
        return output_path
