# ui_elements.py
import tkinter as tk

class UIElements:
    @staticmethod
    def create_labeled_entry(root, label_text, row, command, theme):
        label = tk.Label(root, text=label_text, bg=theme['bg'], fg=theme['fg'])
        label.grid(row=row, column=0, padx=10, pady=10, sticky="e")
        entry = tk.Entry(root, width=50, bg=theme['entry_bg'], fg=theme['entry_fg'])
        entry.grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        browse_button = tk.Button(root, text="Browse...", command=command, bg=theme['button_bg'], fg=theme['button_fg'])
        browse_button.grid(row=row, column=2, padx=10, pady=10)
        return entry

    @staticmethod
    def create_volume_slider(root, theme, set_volume):
        volume_frame = tk.Frame(root, bg=theme['bg'])
        volume_frame.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        volume_slider = tk.Scale(volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume,
                                bg=theme['volume_slider_bg'], fg=theme['volume_slider_fg'],
                                troughcolor=theme['volume_slider_trough'], sliderrelief=tk.FLAT)
        volume_slider.set(100)
        volume_slider.pack(fill=tk.X, padx=5, pady=5)
        return volume_slider
