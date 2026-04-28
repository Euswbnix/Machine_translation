"""Entry point for the GUI app.

Run:
    python -m gui            # from project root (so src/ is importable)
"""

import sys
from PySide6.QtWidgets import QApplication

from gui.translator import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Transformer MT")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
