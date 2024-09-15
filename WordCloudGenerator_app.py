import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import os
import threading
from WordCloudGenerator import WordCloudGenerator

class GenerateWordCloudApp:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme
        self.icon_path = "icon.ico"
        self.generator = WordCloudGenerator(theme, self.update_status)
        self.window = tk.Toplevel(master)
        self.window.title("Generate Word Cloud")
        self.setup_ui()
        self.set_icon()

    def setup_ui(self):
        self.window.configure(bg=self.theme['bg'])
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=3)
        self.window.columnconfigure(2, weight=1)
        
        self.label = tk.Label(self.window, text="Select JSONL File:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky='e')
        
        self.entry_file = tk.Entry(self.window, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'])
        self.entry_file.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

        self.browse_button = tk.Button(self.window, text="Browse", command=self.select_file, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.generate_button = tk.Button(self.window, text="Generate Word Cloud", command=self.start_wordcloud_generation, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.generate_button.grid(row=1, column=0, columnspan=3, pady=20)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.window, textvariable=self.status_var, anchor='w', relief=tk.SUNKEN, bg=self.theme['bg'], fg=self.theme['fg'])
        self.status_label.grid(row=2, column=0, columnspan=3, sticky='ew')

    def set_icon(self):
        if os.path.exists(self.icon_path):
            self.window.iconbitmap(self.icon_path)
        else:
            print("Icon file not found.")

    def select_file(self):
        filetypes = [("JSON Lines files", "*.jsonl")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, file_path)

    def start_wordcloud_generation(self):
        file_path = self.entry_file.get()
        if os.path.isfile(file_path):
            # Run the word cloud generation in a background thread
            threading.Thread(target=self.generate_wordcloud, args=(file_path,), daemon=True).start()
        else:
            self.update_status("Input Error: The file does not exist.")

    def generate_wordcloud(self, file_path):
        img_tk = self.generator.generate_wordcloud(file_path)
        if img_tk:
            self.window.after(0, self.show_image, img_tk)

    def show_image(self, img_tk):
        top = tk.Toplevel(self.window)
        top.title("Word Cloud")
        top.geometry("616x308")
        top.configure(bg=self.theme['bg'])
        if os.path.exists(self.icon_path):
            top.iconbitmap(self.icon_path)
        
        frame = tk.Frame(top, bg=self.theme['bg'])
        frame.pack(fill=tk.BOTH, expand=True)
        
        label = tk.Label(frame, image=img_tk, bg=self.theme['bg'])
        label.image = img_tk
        label.pack()

    def update_status(self, message):
        self.window.after(0, self.status_var.set, message)

if __name__ == "__main__":
    root = tk.Tk()

    # Define a simple theme
    theme = {
        'bg': '#f0f0f0',
        'fg': '#000000',
        'entry_bg': '#ffffff',
        'entry_fg': '#000000',
        'button_bg': '#007bff',
        'button_fg': '#ffffff'
    }

    app = GenerateWordCloudApp(root, theme)
    root.mainloop()
