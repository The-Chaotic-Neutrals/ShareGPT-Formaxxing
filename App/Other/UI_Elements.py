import tkinter as tk

class UIElements:
    @staticmethod
    def create_labeled_entry(root, label_text, row, command, theme):
        frame = tk.Frame(root, bg=theme['bg'])
        frame.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        label = tk.Label(frame, text=label_text, bg=theme['bg'], fg=theme['fg'])
        label.pack(side=tk.LEFT)
        
        entry = tk.Entry(frame, width=50, bg=theme['entry_bg'], fg=theme['entry_fg'])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_button = tk.Button(frame, text="Browse...", command=command, bg=theme['button_bg'], fg=theme['button_fg'])
        browse_button.pack(side=tk.RIGHT, padx=10)
        
        return entry

    @staticmethod
    def create_volume_slider(parent, theme, set_volume):
        volume_frame = tk.Frame(parent, bg=theme['bg'])
        volume_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        volume_slider = tk.Scale(volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume,
                                bg=theme['volume_slider_bg'], fg=theme['volume_slider_fg'],
                                troughcolor=theme['volume_slider_trough'], sliderrelief=tk.FLAT)
        volume_slider.set(100)
        volume_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        return volume_slider
