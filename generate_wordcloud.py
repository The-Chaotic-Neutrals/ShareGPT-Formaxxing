import tkinter as tk
from tkinter import filedialog
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import io
import json
import os
import threading
import re

class GenerateWordCloudApp:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme
        self.icon_path = "icon.ico"
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
        self.update_status("Loading file...")
        try:
            text = self.load_and_process_text(file_path)
            if not text.strip():
                self.update_status("No text found in the file to generate a word cloud.")
                return

            self.update_status("Generating word cloud...")
            wordcloud = WordCloud(width=616, height=308, background_color='#2e2e2e').generate(text)
            
            # Save the word cloud to an image stream
            image_stream = io.BytesIO()
            self.save_wordcloud_image(wordcloud, image_stream)
            
            # Save the image to file
            output_dir = "wordclouds"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "wordcloud.png")
            image_stream.seek(0)
            img = Image.open(image_stream)
            img.save(output_path)
            
            # Convert image to Tkinter format
            img_tk = ImageTk.PhotoImage(img)

            # Update GUI with the generated image and status
            self.window.after(0, self.show_image, img_tk)
            self.update_status(f"Word cloud generated and saved to {output_path}.")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def save_wordcloud_image(self, wordcloud, image_stream):
        """ Save the word cloud image to the given stream """
        # Ensure figure creation and saving happens in the main thread
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(image_stream, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def load_and_process_text(self, file_path):
        text = ''
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    if 'conversations' in data:
                        for message in data['conversations']:
                            if message.get('from') != 'system' and 'value' in message:
                                value = re.sub(r'[^\w\s]', '', message['value'].lower())
                                text += ' ' + value
        except Exception as e:
            self.update_status(f"Error processing file: {str(e)}")
        return text

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
