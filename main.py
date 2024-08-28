# main.py
import tkinter as tk
from ui_manager import UIManager

def main():
    root = tk.Tk()
    app = UIManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()
