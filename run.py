from time import time
start = time()
import settings
from PyQt5.QtWidgets import QApplication
from app.MainWindow import MainWindow
end = time()
print(f"Library loading time: {end-start}")


if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    end2 = time()
    print(f"Application startup time: {end2-end}")
    app.exec_()
