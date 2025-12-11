from App.Other.BG import FloatingPixelsWidget

class Theme:
    DARK = {
        'bg': '#000000',                # pitch black background
        'fg': '#1e90ff',                # dodger blue foreground (buttons, highlights)
        'text_bg': '#000000',           # black text background
        'text_fg': '#ffffff',           # pure white text for maximum contrast
        'entry_bg': '#000000',          # black for input backgrounds
        'entry_fg': '#ffffff',          # white text inside inputs
        'button_bg': '#1e90ff',         # bright dodger blue buttons
        'button_fg': '#ffffff',         # white button text
        'background_widget_class': FloatingPixelsWidget,  # Reference to animated pixel background from bg.py
    }