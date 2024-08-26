Launch with python Formaxxing.py either in cmd/terminal or via the provided bat file.
Preview window shows the first 3 entries processed and converted.
Basic debugging, but needs work.
PR's Welcome.

Modularity Update:
main.py is now the entry point and only responsible for creating the UI.
dataset_converter.py contains all the logic for data loading and processing, without any UI-related code.
ui_manager.py handles all the UI-related code and Pygame initialization, using the DatasetConverter class for the actual data processing.