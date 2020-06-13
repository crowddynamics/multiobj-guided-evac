import os
from qtgui.cli import run_gui

if __name__ == '__main__':
    # Full (absolute) path to architecture file
    root = os.path.dirname(__file__)
    run_gui(os.path.join(root, "complex_variance.py"))

