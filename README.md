Launch with python Main.py either in cmd/terminal or via the provided bat/sh file.

PR's Welcome.

Datamaxxer CLI - info below.
| **Argument**                  | **Description**                                                                                   | **Type**                | **Default**      |
|-------------------------------|---------------------------------------------------------------------------------------------------|-------------------------|------------------|
| `input_file`                   | Path to the input JSONL file containing conversations to be filtered.                             | `str`                   | Required         |
| `output_dir`                   | Path to the directory where the filtered conversations will be saved.                             | `str`                   | Required         |
| `--check_blank_turns`          | Enable filtering for blank turns (messages without a value or empty system role).                 | `store_true` (boolean)  | `True`           |
| `--check_invalid_endings`      | Enable filtering for conversations with invalid endings (messages ending with a letter, number, or comma). | `store_true` (boolean)  | `True`           |
| `--check_null_gpt`             | Enable filtering for conversations where GPT responses are null (missing value).                 | `store_true` (boolean)  | `True`           |
| `--check_duplicate_system`     | Enable filtering for duplicate system messages (system messages followed by human responses with identical content). | `store_true` (boolean)  | `True`           |
| `--allow_empty_system_role`    | Allow conversations where the system role has an empty or null value.                             | `store_true` (boolean)  | `True`           |
