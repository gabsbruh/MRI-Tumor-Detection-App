import time
start = time.time()
import settings
from PyQt5.QtWidgets import QApplication
from app.MainWindow import MainWindow
end = time.time()
print(f"zaladowanie bibliotek: {end-start}")

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    end2 = time.time()
    print(f"wyswietlenie apki: {end2-end}")
    app.exec_()
