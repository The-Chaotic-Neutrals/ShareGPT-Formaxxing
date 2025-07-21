import tkinter as tk
from tkinter import ttk, messagebox
from music_player import MusicPlayer
import os

class MusicPlayerApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.music_player = MusicPlayer()
        self.volume_file = "volume.txt"  # File to store volume settings
        self.setup_ui()
        self.load_volume()

    def setup_ui(self):
        self.window = tk.Toplevel(self.root)
        self.window.title("MusicMaxxer")
        self.window.configure(bg=self.theme.get('bg', 'white'))

        # Set the window icon
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)

        self.music_button = tk.Button(self.window, text="Play Music", command=self.play_music, bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
        self.music_button.pack(pady=10, padx=5)

        volume_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        volume_frame.pack(pady=10, padx=5, fill='x')

        self.volume_slider = ttk.Scale(volume_frame, from_=0, to=100, orient='horizontal', command=self.set_volume)
        self.volume_slider.pack(fill='x', padx=10)
        
    def play_music(self):
        try:
            status = self.music_player.play_music()
            self.music_button.config(text=status)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play music: {e}")
            self.music_button.config(text="Play Music")

    def set_volume(self, value):
        self.music_player.set_volume(float(value))
        self.save_volume(float(value))

    def load_volume(self):
        # Load the saved volume from a file, if it exists
        if os.path.exists(self.volume_file):
            with open(self.volume_file, "r") as file:
                volume = file.read().strip()
                try:
                    volume = float(volume)
                    self.volume_slider.set(volume * 100)  # Scale value is between 0 and 100
                except ValueError:
                    self.volume_slider.set(7)  # Default volume if file is corrupted

    def save_volume(self, volume):
        # Save the volume to a file
        with open(self.volume_file, "w") as file:
            file.write(f"{volume}")
