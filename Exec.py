import sys
import MainWindow
from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow =  MainWindow.MainWindow()
    sys.exit(app.exec_())