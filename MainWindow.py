import sys, cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit , QWidget, QGridLayout, \
    QGroupBox, QPushButton, QVBoxLayout, QFileDialog
from ExtendedButton import ExtendedQPushButton

class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.btnSet = []
        self.pic = None
        self.legal = [0, 0, 0, 0, 0]

        self.show()
        self.initUI() 
        self.setWindowTitle("2020 Opencvdl HW1")
        self.resize(600, 300)

    def initUI(self):

        #first
        self.loadImgBtn = QPushButton('1.1 Load image')
        self.ColorSepBtn = QPushButton('1.2 Color seperation')
        self.imgFlipBtn = QPushButton('1.3 Image Flipping')
        self.BlendBtn = QPushButton('1.4 Blending')

        #second
        self.midFilterBtn = QPushButton('2.1 Medium Filter')
        self.gauBlur2Btn = QPushButton('2.2 Gaussian Blur')
        self.bilateralBtn = QPushButton('2.3 Bilateral')

        #third
        self.gauBlur3Btn = QPushButton('3.1Gaussion Blur')
        self.sobelXBtn = QPushButton('3.2 Sobel X')
        self.sobelYBtn = QPushButton('3.3 Sobel Y')
        self.magnitudeBtn = QPushButton('3.4 Magnitude')

        #last
        self.rotateLabel = QLabel('Rotation:', self)
        self.scaleLabel = QLabel('Scaling:', self)
        self.TxLabel = QLabel('Tx:', self)
        self.TyLabel = QLabel('Ty:', self)

        self.rotateText = QLineEdit(self)
        self.scaleText = QLineEdit(self)
        self.TxText = QLineEdit(self)
        self.TyText = QLineEdit(self)

        self.rotUnitLabel = QLabel('deg', self)
        self.TxUnitLabel = QLabel('pixel', self)
        self.TyUnitLabel = QLabel('pixel', self)

        self.transformBtn = QPushButton('4.Transformation', self)


        grid = QGridLayout()
        gridTitle = ['1.Image Processing', '2.Image Smoothing', '3.Edge Detection', '4.Transformation']
        
        btnIds = [['1.1', '1.2', '1.3', '1.4'],
        ['2.1', '2.2', '2.3'],
        ['3.1', '3.2', '3.3', '3.4']]
        grid.addWidget(self.createThreeGroup([self.loadImgBtn, self.ColorSepBtn, self.imgFlipBtn, self.BlendBtn], \
            gridTitle[0], btnIds[0]), 0, 0)
        grid.addWidget(self.createThreeGroup([self.midFilterBtn, self.gauBlur2Btn, self.bilateralBtn], \
            gridTitle[0], btnIds[0]), 0, 1)
        grid.addWidget(self.createThreeGroup([self.gauBlur3Btn,  self.sobelXBtn, self.sobelYBtn, self.magnitudeBtn], \
            gridTitle[0], btnIds[0]), 0, 2)
        grid.addWidget(self.createLastGroup(gridTitle[3]), 0, 3)
        self.setLayout(grid)

    def createThreeGroup(self, button, title, ids):
        groupBox = QGroupBox(title)

        for i in range(len(button)):
            button[i].id = ids[i]
            button[i].clicked.connect(self.btnClickedHandler)

        vbox = QVBoxLayout()
        for i in range(len(button)):
            vbox.addWidget(button[i])
        groupBox.setLayout(vbox)

        return groupBox

    def createLastGroup(self, title):
        groupBox = QGroupBox(title)

        grid = QGridLayout()
        grid.addWidget(self.rotateLabel, 0, 0, 1, 1)
        grid.addWidget(self.scaleLabel, 1, 0, 1, 1)
        grid.addWidget(self.TxLabel, 2, 0, 1, 1)
        grid.addWidget(self.TyLabel, 3, 0, 1, 1)
        grid.addWidget(self.rotateText, 0, 1, 1, 1)
        grid.addWidget(self.scaleText, 1, 1, 1, 1)
        grid.addWidget(self.TxText, 2, 1, 1, 1)
        grid.addWidget(self.TyText, 3, 1, 1, 1)
        grid.addWidget(self.rotUnitLabel, 0, 2, 1, 1)
        grid.addWidget(self.TxUnitLabel, 2, 2, 1, 1)
        grid.addWidget(self.TyUnitLabel, 3, 2, 1, 1)
        grid.addWidget(self.transformBtn, 4, 0, 1, 3)
        groupBox.setLayout(grid)

        return groupBox

    def btnClickedHandler(self):
        btn = self.sender()
        bid = btn.id
        if bid == '1.1':
            picFName, fileType = QFileDialog.getOpenFileName(self,"選取檔案","./","Picture file (*.png *.jpg *.jpeg)")
            picFileType = picFName.split('.')[-1]
            if picFileType == 'jpg' or picFileType == 'png' or picFileType == 'jpeg':
                self.legal[0] = 1 # enable 1.2
                print(picFName)
                self.pic = cv2.imread(picFName)
                print('Height : ' + str(self.pic.shape[0]))
                print('Width : ' + str(self.pic.shape[1]))
                cv2.imshow('My Image', self.pic)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif bid == '1.2' and self.legal[0] > 0:
            self.legal[0] = 2 
            z = np.zeros((self.pic.shape[0], self.pic.shape[1]), dtype=self.pic.dtype)
            b = np.zeros((self.pic.shape[0], self.pic.shape[1]), dtype=self.pic.dtype)
            g = np.zeros((self.pic.shape[0], self.pic.shape[1]), dtype=self.pic.dtype)
            r = np.zeros((self.pic.shape[0], self.pic.shape[1]), dtype=self.pic.dtype)
            b[:,:] = self.pic[:,:,0]
            g[:,:] = self.pic[:,:,1]
            r[:,:] = self.pic[:,:,2]
            bsep = cv2.merge([b, z, z])
            gsep = cv2.merge([z, g, z])
            rsep = cv2.merge([z, z, r])
            rgbConcate = np.hstack((rsep, gsep, bsep))
            print(rgbConcate.shape)
            ratio1 = rgbConcate.shape[1] / 1200
            ratio2 = self.pic.shape[1] / 600
            rgbConcate = cv2.resize(rgbConcate, (int(rgbConcate.shape[1]/ratio1), int(rgbConcate.shape[0]/ratio1)))
            resizedPic = cv2.resize(self.pic, (int(self.pic.shape[1]/ratio2), int(self.pic.shape[0]/ratio2)))
            cv2.imshow("original", resizedPic)
            cv2.imshow("Seperate", rgbConcate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
        elif bid == '1.3' and self.legal[0] > 0:
            self.legal[0] = 2 
            flipping = np.zeros((self.pic.shape[0],self.pic.shape[1],self.pic.shape[2]),dtype=self.pic.dtype)
            wsize = self.pic.shape[1]
            for i in range(wsize):
                flipping[:,i,:] = self.pic[:,wsize - 1 - i,:]
            
            ratio = self.pic.shape[1] / 600
            resizedPic = cv2.resize(self.pic, (int(self.pic.shape[1]/ratio), int(self.pic.shape[0]/ratio)))
            resizedFlipping = cv2.resize(flipping, (int(flipping.shape[1]/ratio), int(flipping.shape[0]/ratio)))
            cv2.imshow("original", resizedPic)
            cv2.imshow("filpping", resizedFlipping)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(app.exec_())