import pygame
import os
from tkinter import messagebox

class MusicPlayer:
    def __init__(self):
        try:
            pygame.init()
            pygame.mixer.init()
            self.mp3_file_path = "kitchen.mp3"
        except Exception as e:
            messagebox.showerror("Pygame Initialization Error", f"An error occurred while initializing Pygame: {str(e)}")

    def play_music(self):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                return "Play Music"
            else:
                if os.path.exists(self.mp3_file_path):
                    pygame.mixer.music.load(self.mp3_file_path)
                    pygame.mixer.music.play()
                    return "Pause Music"
                else:
                    messagebox.showerror("File Not Found", "The music file does not exist.")
                    return "Play Music"
        except Exception as e:
            messagebox.showerror("Playback Error", f"An error occurred while playing the music: {str(e)}")
            return "Play Music"

    def set_volume(self, value):
        try:
            volume = int(value) / 100
            pygame.mixer.music.set_volume(volume)
        except ValueError:
            pass
