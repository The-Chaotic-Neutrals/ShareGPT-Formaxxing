import os
import json
import random

class LineMancerCore:
    def split_jsonl(self, input_path, output_dir, prefix, lines_per_file):
        if not os.path.isfile(input_path):
            raise FileNotFoundError("Invalid input file.")
        if not os.path.isdir(output_dir):
            raise NotADirectoryError("Invalid output directory.")
        if not prefix:
            raise ValueError("Prefix required.")
        if lines_per_file <= 0:
            raise ValueError("Lines per file must be > 0.")

        index, buffer = 1, []
        with open(input_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        buffer.append(line)
                    except json.JSONDecodeError:
                        print(f"[Warning] Skipping invalid line {i}")
                if len(buffer) >= lines_per_file:
                    self._write_split_file(output_dir, prefix, index, buffer)
                    index += 1
                    buffer = []
            if buffer:
                self._write_split_file(output_dir, prefix, index, buffer)
        return index

    def _write_split_file(self, directory, prefix, index, buffer):
        out_path = os.path.join(directory, f"{prefix}_{index}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(buffer) + "\n")
        print(f"[LineMancer] Wrote {len(buffer)} lines to {out_path}")

    def merge_jsonl(self, input_paths, output_dir, output_filename):
        if not input_paths:
            raise ValueError("No input files provided.")
        if not os.path.isdir(output_dir):
            raise NotADirectoryError("Invalid output directory.")
        if not output_filename.endswith(".jsonl"):
            output_filename += ".jsonl"

        output_path = os.path.join(output_dir, output_filename)
        total = 0
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for fpath in input_paths:
                with open(fpath, 'r', encoding='utf-8') as infile:
                    for i, line in enumerate(infile, 1):
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                                out_file.write(line + "\n")
                                total += 1
                            except json.JSONDecodeError:
                                print(f"[Warning] Skipped bad line in {fpath} at {i}")
        return output_path, total

    def shuffle_jsonl(self, input_path, output_dir):
        if not os.path.isfile(input_path):
            raise FileNotFoundError("Invalid input file.")
        if not os.path.isdir(output_dir):
            raise NotADirectoryError("Invalid output directory.")

        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + "_shuffled.jsonl")

        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        random.shuffle(lines)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in lines:
                obj = json.loads(line)
                json.dump(obj, f_out, ensure_ascii=False)
                f_out.write('\n')
        return output_path
