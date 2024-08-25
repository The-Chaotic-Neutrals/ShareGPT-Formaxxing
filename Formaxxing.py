import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import sqlite3
import csv

def process_plaintext_line(line, debug):
    conversations = []
    
    if 'system:' in line:
        conversations.append({"from": "system", "value": line.split('system:')[1].strip()})
    if 'user:' in line:
        conversations.append({"from": "human", "value": line.split('user:')[1].strip()})
    if 'assistant:' in line:
        conversations.append({"from": "gpt", "value": line.split('assistant:')[1].strip()})
    
    return conversations

def extract_conversations(entry, debug):
    conversations = []
    
    if 'conversations' in entry:
        for message in entry['conversations']:
            role = message.get('from')
            if role == 'system':
                conversations.append({"from": "system", "value": message.get('value', '')})
            elif role == 'user':
                conversations.append({"from": "human", "value": message.get('value', '')})
            elif role == 'gpt':
                conversations.append({"from": "gpt", "value": message.get('value', '')})
    else:
        if 'system' in entry:
            conversations.append({"from": "system", "value": entry['system']})

        if 'completion' in entry:
            if isinstance(entry['completion'], list):
                for message in entry['completion']:
                    role = message.get('role')
                    if role == 'user':
                        conversations.append({"from": "human", "value": message.get('content', '')})
                    elif role == 'assistant':
                        conversations.append({"from": "gpt", "value": message.get('content', '')})
            elif isinstance(entry['completion'], str):
                try:
                    completion = json.loads(entry['completion'])
                    if isinstance(completion, list):
                        for message in completion:
                            role = message.get('role')
                            if role == 'user':
                                conversations.append({"from": "human", "value": message.get('content', '')})
                            elif role == 'assistant':
                                conversations.append({"from": "gpt", "value": message.get('content', '')})
                except json.JSONDecodeError:
                    if debug:
                        print(f"Invalid JSON in completion field: {entry['completion']}")

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
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            if debug:
                                print(f"Skipped invalid JSON line: {line.strip()}")
        elif input_path.endswith('.parquet'):
            data = pd.read_parquet(input_path).to_dict(orient='records')
        elif input_path.endswith('.txt'):
            data = []
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.JSONDecodeError:
                            conversations = process_plaintext_line(line, debug)
                            if conversations:
                                data.append({"conversations": conversations})
                            if debug:
                                print(f"Skipped invalid JSON line: {line.strip()}")
        elif input_path.endswith('.csv'):
            data = []
            with open(input_path, 'r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    entry = {}
                    if 'system' in row:
                        entry['system'] = row['system']
                    if 'completion' in row:
                        entry['completion'] = row['completion']
                    data.append(entry)
        elif input_path.endswith('.sql'):
            connection = sqlite3.connect(":memory:")
            cursor = connection.cursor()
            with open(input_path, 'r', encoding='utf-8') as infile:
                sql_script = infile.read()
            cursor.executescript(sql_script)
            data = cursor.execute("SELECT * FROM data").fetchall()
            column_names = [description[0] for description in cursor.description]
            data = [dict(zip(column_names, row)) for row in data]
            connection.close()
        else:
            raise ValueError("Unsupported file format. Please use .json, .jsonl, .parquet, .txt, .csv, or .sql files.")

        preview_entries = []
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for entry in data:
                conversations = extract_conversations(entry, debug)
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
        
        status_var.set("Conversion completed successfully!")

    except json.JSONDecodeError as e:
        messagebox.showerror("Error", f"Failed to decode JSON: {str(e)}")
    except IOError:
        messagebox.showerror("Error", "Failed to read/write file. Check file paths and permissions.")
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
    except sqlite3.Error as se:
        messagebox.showerror("Error", f"SQLite error: {str(se)}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[
        ("JSON files", "*.json"),
        ("JSON Lines files", "*.jsonl"),
        ("Parquet files", "*.parquet"),
        ("Plaintext files", "*.txt"),
        ("CSV files", "*.csv"),
        ("SQL files", "*.sql")
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

def toggle_dark_mode():
    if dark_mode_var.get():
        apply_dark_mode()
    else:
        apply_light_mode()

def apply_dark_mode():
    root.tk_setPalette(background='#2e2e2e', foreground='#ffffff')
    style.configure('TButton', background='#3e3e3e', foreground='#ffffff', padding=6)
    style.configure('TLabel', background='#2e2e2e', foreground='#ffffff')
    style.configure('TCheckbutton', background='#2e2e2e', foreground='#ffffff')
    style.configure('TEntry', background='#3e3e3e', foreground='#ffffff')
    style.configure('TScrolledText', background='#3e3e3e', foreground='#ffffff')
    status_bar.config(bg='#2e2e2e', fg='#ffffff')

def apply_light_mode():
    root.tk_setPalette(background='#ffffff', foreground='#000000')
    style.configure('TButton', background='#f0f0f0', foreground='#000000', padding=6)
    style.configure('TLabel', background='#ffffff', foreground='#000000')
    style.configure('TCheckbutton', background='#ffffff', foreground='#000000')
    style.configure('TEntry', background='#f0f0f0', foreground='#000000')
    style.configure('TScrolledText', background='#f0f0f0', foreground='#000000')
    status_bar.config(bg='#ffffff', fg='#000000')

# Set up the Tkinter window
root = tk.Tk()
root.title("Dataset Converter")

# Load and set the window icon
icon_path = "./icon.png"  # Change this to the path of your icon file
try:
    icon = tk.PhotoImage(file=icon_path)
    root.iconphoto(False, icon)
except tk.TclError:
    messagebox.showerror("Error", "Failed to load icon. Ensure the file path and format are correct.")

# Set up style
style = ttk.Style()

# Default to light mode
dark_mode_var = tk.BooleanVar(value=False)

# Configure the grid layout
root.columnconfigure(1, weight=1)
root.rowconfigure(6, weight=1)

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

# Dark Mode Toggle Button
tk.Checkbutton(root, text="Dark Mode", variable=dark_mode_var, command=toggle_dark_mode).grid(row=4, column=0, columnspan=3, pady=10)

# Preview window
tk.Label(root, text="Preview Output:").grid(row=5, column=0, padx=10, pady=10, sticky="nw")
preview_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=('Consolas', 10))
preview_text.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

# Status Bar
status_var = tk.StringVar()
status_bar = tk.Label(root, textvariable=status_var, anchor='w', relief=tk.SUNKEN)
status_bar.grid(row=7, column=0, columnspan=3, sticky='ew')

# Set up resizing behavior
root.grid_rowconfigure(6, weight=1)
root.grid_columnconfigure(1, weight=1)

def pause():
    root.quit()
    messagebox.showinfo("Info", "Press OK to exit.")

root.protocol("WM_DELETE_WINDOW", pause)
root.mainloop()