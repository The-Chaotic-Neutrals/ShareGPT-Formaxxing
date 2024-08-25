import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

def process_plaintext_line(line, debug):
    conversations = []
    
    if 'system:' in line:
        conversations.append({"from": "system", "value": line.split('system:')[1].strip()})
    if 'user:' in line:
        conversations.append({"from": "human", "value": line.split('user:')[1].strip()})
    if 'assistant:' in line:
        conversations.append({"from": "gpt", "value": line.split('assistant:')[1].strip()})
    
    return conversations

def convert_dataset(input_path, output_path, debug):
    try:
        if input_path.endswith('.json'):
            with open(input_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
        elif input_path.endswith('.jsonl'):
            data = []
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data.append(json.loads(line))
        elif input_path.endswith('.parquet'):
            data = pd.read_parquet(input_path).to_dict(orient='records')
        elif input_path.endswith('.csv'):
            data = pd.read_csv(input_path).to_dict(orient='records')
        elif input_path.endswith('.txt'):
            data = []
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.JSONDecodeError:
                        conversations = process_plaintext_line(line, debug)
                        if conversations:
                            data.append({"conversations": conversations})
                        if debug:
                            print(f"Skipped invalid JSON line: {line.strip()}")
        else:
            raise ValueError("Unsupported file format. Please use .json, .jsonl, .parquet, .csv, or .txt files.")

        preview_entries = []
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for entry in data:
                conversations = []

                # Handle system message first
                if 'system' in entry:
                    conversations.append({"from": "system", "value": entry['system']})

                # Handle JSONL-specific format
                if 'completion' in entry:
                    for message in entry['completion']:
                        if message['role'] == 'user':
                            conversations.append({"from": "human", "value": message['content']})
                        elif message['role'] == 'assistant':
                            conversations.append({"from": "gpt", "value": message['content']})

                preview_output = {"conversations": conversations}
                if len(preview_entries) < 3:
                    preview_entries.append(preview_output)

                if conversations:
                    outfile.write(json.dumps({"conversations": conversations}) + '\n')
                elif debug:
                    print("No relevant messages found in entry.")
        
        preview_text.delete(1.0, tk.END)
        if preview_entries:
            preview_text.insert(tk.END, json.dumps(preview_entries, indent=2))
        else:
            preview_text.insert(tk.END, "No conversations found for this dataset.")
        
        messagebox.showinfo("Success", "Conversion completed successfully!")

    except json.JSONDecodeError as e:
        messagebox.showerror("Error", f"Failed to decode JSON: {str(e)}")
    except IOError:
        messagebox.showerror("Error", "Failed to read/write file. Check file paths and permissions.")
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[
        ("JSON files", "*.json"),
        ("JSON Lines files", "*.jsonl"),
        ("Parquet files", "*.parquet"),
        ("CSV files", "*.csv"),
        ("Plaintext files", "*.txt")
    ])
    if file_path:
        entry_input_file.delete(0, tk.END)
        entry_input_file.insert(0, file_path)

def select_output_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=[("JSON Lines files", "*.jsonl")])
    if file_path:
        entry_output_file.delete(0, tk.END)
        entry_output_file.insert(0, file_path)

def on_convert_button_click():
    input_path = entry_input_file.get()
    output_path = entry_output_file.get()
    debug = debug_var.get()
    if input_path and output_path:
        convert_dataset(input_path, output_path, debug)
    else:
        messagebox.showwarning("Input Error", "Please select both input and output files.")

def apply_dark_mode():
    root.tk_setPalette(background='#2e2e2e', foreground='#ffffff')
    style.configure('TButton', background='#3e3e3e', foreground='#ffffff', padding=6)
    style.configure('TLabel', background='#2e2e2e', foreground='#ffffff')
    style.configure('TCheckbutton', background='#2e2e2e', foreground='#ffffff')
    style.configure('TEntry', background='#3e3e3e', foreground='#ffffff')
    style.configure('TScrolledText', background='#3e3e3e', foreground='#ffffff')

# Set up the Tkinter window
root = tk.Tk()
root.title("Dataset Converter")

# Apply dark mode
style = ttk.Style()
apply_dark_mode()

# Configure the grid layout
root.columnconfigure(1, weight=1)
root.rowconfigure(4, weight=1)

# Input file selection
tk.Label(root, text="Input File:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
entry_input_file = tk.Entry(root, width=50)
entry_input_file.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
tk.Button(root, text="Browse...", command=select_input_file).grid(row=0, column=2, padx=10, pady=10)

# Output file selection
tk.Label(root, text="Output File:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
entry_output_file = tk.Entry(root, width=50)
entry_output_file.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
tk.Button(root, text="Browse...", command=select_output_file).grid(row=1, column=2, padx=10, pady=10)

# Debugging option
debug_var = tk.BooleanVar()
tk.Checkbutton(root, text="Enable Debugging", variable=debug_var).grid(row=2, column=0, columnspan=3, pady=10)

# Convert button
tk.Button(root, text="Convert", command=on_convert_button_click).grid(row=3, column=0, columnspan=3, pady=20)

# Preview window
tk.Label(root, text="Preview Output:").grid(row=4, column=0, padx=10, pady=10, sticky="n")
preview_text = scrolledtext.ScrolledText(root, wrap=tk.WORD)
preview_text.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")

# Set up resizing behavior
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(1, weight=1)

# Add a pause at the end of the script
def pause():
    root.withdraw()
    messagebox.showinfo("Info", "Press OK to exit.")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", pause)
root.mainloop()
