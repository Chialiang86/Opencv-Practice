import sys

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QGroupBox, QPushButton, QVBoxLayout

class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi()
        self.show()
        grid = QGridLayout()
        btnName1 = ['1.1 Load image', '1.2 Color seperation', '1.3 Image Flipping', '1.4 Blending']
        btnName2 = ['2.1 Medium Filter', '2.2 Gaussian Blur', '2.3 Bilateral']
        btnName3 = ['3.1Gaussion Blur', '3.2 Sobel X', '3.3 Sobel Y', '3.4 Magnitude']
        gridTitle = ['1.Image Processing', '2.Image Smoothing', '3.Edge Detection', '4.Transformation']
        grid.addWidget(self.createExample_Group(4, btnName1), 0, 0)
        grid.addWidget(self.createExample_Group(3, btnName2), 0, 1)
        grid.addWidget(self.createExample_Group(4, btnName3), 0, 2)
        self.setLayout(grid)

        self.setWindowTitle("PyQt5 Group Box")
        self.resize(400, 300)

    def createExample_Group(self, num, btnName = []):
        groupBox = QGroupBox("Best Food")

        button = []
        for i in range(num):
            button.append(QPushButton(btnName[i]))

        vbox = QVBoxLayout()
        for i in range(num):
            vbox.addWidget(button[i])
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def setupUi(self):
        self.setWindowTitle("Hello World!")

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(app.exec_())