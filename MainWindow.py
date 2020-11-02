import sys, cv2
import numpy as np
import matplotlib._mathtext_data
import math

from scipy import signal
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit , QWidget, QGridLayout, \
    QGroupBox, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QSpinBox
from ExtendedButton import ExtendedQPushButton
from VGG16 import VGG

class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.btnSet = []
        self.pic = None
        self.legal = [0] * 5

        self.__imgCache = [0] * 8
        self.__imgCenter = np.array([160, 84])
        self.__model = 0

        self.show()
        self.initUI() 
        self.setWindowTitle("2020 Opencvdl HW1")
        self.resize(800, 400)

        self.__msg = QMessageBox()
        self.__msg.setIcon(QMessageBox.Warning)
        self.__msg.setWindowTitle("Oops!!!!")

    def initUI(self):

        #first
        self.__loadImgBtn = ExtendedQPushButton('1.1 Load image')
        self.__ColorSepBtn = ExtendedQPushButton('1.2 Color seperation')
        self.__imgFlipBtn = ExtendedQPushButton('1.3 Image Flipping')
        self.__BlendBtn = ExtendedQPushButton('1.4 Blending')

        #second
        self.__midFilterBtn = ExtendedQPushButton('2.1 Medium Filter')
        self.__gauBlur2Btn = ExtendedQPushButton('2.2 Gaussian Blur')
        self.__bilateralBtn = ExtendedQPushButton('2.3 Bilateral')

        #third
        self.__gauBlur3Btn = ExtendedQPushButton('3.1Gaussion Blur')
        self.__sobelXBtn = ExtendedQPushButton('3.2 Sobel X')
        self.__sobelYBtn = ExtendedQPushButton('3.3 Sobel Y')
        self.__magnitudeBtn = ExtendedQPushButton('3.4 Magnitude')

        #fourth
        self.__rotateLabel = QLabel('Rotation:', self)
        self.__scaleLabel = QLabel('Scaling:', self)
        self.__TxLabel = QLabel('Tx:', self)
        self.__TyLabel = QLabel('Ty:', self)

        self.__rotateText = QLineEdit('30', self)
        self.__scaleText = QLineEdit('0.9', self)
        self.__TxText = QLineEdit('200', self)
        self.__TyText = QLineEdit('300', self)

        self.__rotUnitLabel = QLabel('deg', self)
        self.__TxUnitLabel = QLabel('pixel', self)
        self.__TyUnitLabel = QLabel('pixel', self)

        self.__transformBtn = ExtendedQPushButton('4.Transformation')
        self.__transformBtn.id = '4.1'
        self.__transformBtn.clicked.connect(self.btnClickedHandler)

        #fifth
        self.__showImgBtn = ExtendedQPushButton('1.Show Train Images')
        self.__showHyperParamBtn = ExtendedQPushButton('2.Show Hyperparameters')
        self.__showModelStrucBtn = ExtendedQPushButton('3.Show Model Structure')
        self.__showAccBtn = ExtendedQPushButton('4.Show Accuracy')
        self.__testImgIndex = QSpinBox()
        self.__testBtn = ExtendedQPushButton('5.test')

        self.__showImgBtn.id = '5.1'
        self.__showHyperParamBtn.id = '5.2'
        self.__showModelStrucBtn.id = '5.3'
        self.__showAccBtn.id = '5.4'
        self.__testBtn.id = '5.5'

        self.__testImgIndex.setMinimum(0)
        self.__testImgIndex.setMaximum(9999)
        self.__testImgIndex.setValue(1000)

        self.__showImgBtn.clicked.connect(self.btnClickedHandler)
        self.__showHyperParamBtn.clicked.connect(self.btnClickedHandler)
        self.__showModelStrucBtn.clicked.connect(self.btnClickedHandler)
        self.__showAccBtn.clicked.connect(self.btnClickedHandler)
        self.__testBtn.clicked.connect(self.btnClickedHandler)

        grid = QGridLayout()
        gridTitle = ['1.Image Processing', '2.Image Smoothing', '3.Edge Detection', '4.Transformation', '5.VGG16 test']
        
        btnIds = [['1.1', '1.2', '1.3', '1.4'],
        ['2.1', '2.2', '2.3'],
        ['3.1', '3.2', '3.3', '3.4']]
        grid.addWidget(self.createThreeGroup([self.__loadImgBtn, self.__ColorSepBtn, self.__imgFlipBtn, self.__BlendBtn], \
            gridTitle[0], btnIds[0]), 0, 0)
        grid.addWidget(self.createThreeGroup([self.__midFilterBtn, self.__gauBlur2Btn, self.__bilateralBtn], \
            gridTitle[1], btnIds[1]), 0, 1)
        grid.addWidget(self.createThreeGroup([self.__gauBlur3Btn,  self.__sobelXBtn, self.__sobelYBtn, self.__magnitudeBtn], \
            gridTitle[2], btnIds[2]), 0, 2)
        grid.addWidget(self.create4Group(gridTitle[3]), 0, 3)
        grid.addWidget(self.create5Group(gridTitle[4]), 0, 4)
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

    def create4Group(self, title):
        groupBox = QGroupBox(title)
        grid = QGridLayout()
        grid.addWidget(self.__rotateLabel, 0, 0, 1, 1)
        grid.addWidget(self.__scaleLabel, 1, 0, 1, 1)
        grid.addWidget(self.__TxLabel, 2, 0, 1, 1)
        grid.addWidget(self.__TyLabel, 3, 0, 1, 1)
        grid.addWidget(self.__rotateText, 0, 1, 1, 1)
        grid.addWidget(self.__scaleText, 1, 1, 1, 1)
        grid.addWidget(self.__TxText, 2, 1, 1, 1)
        grid.addWidget(self.__TyText, 3, 1, 1, 1)
        grid.addWidget(self.__rotUnitLabel, 0, 2, 1, 1)
        grid.addWidget(self.__TxUnitLabel, 2, 2, 1, 1)
        grid.addWidget(self.__TyUnitLabel, 3, 2, 1, 1)
        grid.addWidget(self.__transformBtn, 4, 0, 1, 3)
        groupBox.setLayout(grid)
        return groupBox

    def create5Group(self, title):
        groupBox = QGroupBox(title)
        grid = QGridLayout()
        grid.addWidget(self.__showImgBtn, 0, 0, 1, 1)
        grid.addWidget(self.__showHyperParamBtn, 1, 0, 1, 1)
        grid.addWidget(self.__showModelStrucBtn, 2, 0, 1, 1)
        grid.addWidget(self.__showAccBtn, 3, 0, 1, 1)
        grid.addWidget(self.__testImgIndex, 4, 0, 1, 1)
        grid.addWidget(self.__testBtn, 5, 0, 1, 1)
        groupBox.setLayout(grid)
        return groupBox

    def btnClickedHandler(self):
        btn = self.sender()
        bid = btn.id
        if bid == '1.1':
            picFName, fileType = QFileDialog.getOpenFileName(self,"選取檔案","./","Picture file (*.JPG *.png *.jpg *.jpeg)")
            picFileType = picFName.split('.')[-1]
            if picFileType == 'jpg' or picFileType == 'png' or picFileType == 'jpeg':
                self.legal = [1, 1, 1, 1, 0]
                self.pic = cv2.imread(picFName)
                self.__imgCenter = np.array([160, 84])
                print('type:' + str(type(self.pic)))
                print('Height : ' + str(self.pic.shape[0]))
                print('Width : ' + str(self.pic.shape[1]))
                cv2.imshow('My Image', self.pic)
                k = cv2.waitKey(0)
                if k > 0:
                    cv2.destroyAllWindows()
        
        elif bid == '1.2':
            if self.legal[0] >= 1:
                self.legal[0] = 2 if self.legal[0] > 2 else self.legal[0]
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
                cv2.imshow("original", self.pic)
                cv2.imshow("red", rsep)
                cv2.imshow("green", gsep)
                cv2.imshow("blue", bsep)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()
    
        elif bid == '1.3':
            if self.legal[0] >= 1:
                self.legal[0] = 3 
                flipping = np.zeros((self.pic.shape[0],self.pic.shape[1],self.pic.shape[2]),dtype=self.pic.dtype)
                wsize = self.pic.shape[1]
                for i in range(wsize):
                    flipping[:,i,:] = self.pic[:,wsize - 1 - i,:]
                self.__imgCache[0] = self.pic
                self.__imgCache[1] = flipping
                cv2.imshow("original", self.pic)
                cv2.imshow("flipping", flipping)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()

        elif bid == '1.4' :
            if self.legal[0] >= 3:
                cv2.namedWindow('blending')
                cv2.createTrackbar ( 'weight' , 'blending' , 0 , 255, self.__blendShow)
                cv2.setTrackbarPos ( 'weight' , 'blending' , 128)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.3 should be done first!")
                self.__msg.exec_()


        # blur
        elif bid == '2.1':
            if self.legal[1] >= 1:
                midBlurImg = cv2.medianBlur(self.pic, 7) 
                cv2.imshow('original', self.pic)
                cv2.imshow('median', midBlurImg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()

        elif bid == '2.2':
            if self.legal[1] >= 1:
                gauBlurImg = cv2.GaussianBlur(self.pic, (5,5), 0)
                cv2.imshow('original', self.pic)
                cv2.imshow('Gaussian', gauBlurImg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()

        elif bid == '2.3':
            if self.legal[1] >= 1:
                bilBlurImg = cv2.bilateralFilter(self.pic, 9, 90, 90)
                cv2.imshow('original', self.pic)
                cv2.imshow('bilateral', bilBlurImg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()

        #edge detect
        elif bid == '3.1':
            if self.legal[2] >= 1:
                self.legal[2] = 2
                grayImg = cv2.cvtColor(self.pic, cv2.COLOR_BGR2GRAY)
                gauKernel = self.__gaussianKer(7)
                gauBlur = signal.convolve2d(grayImg, gauKernel,  boundary='symm', mode='same')
                gauBlur = gauBlur.astype(np.uint8)
                self.__imgCache[0] = gauBlur
                cv2.imshow('original', grayImg)
                cv2.imshow('Gaussian', gauBlur)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()

        elif bid == '3.2' :
            if self.legal[2] >= 2:
                self.legal[2] = 4 if self.legal[2] == 3 else 3
                sobelKernel = self.__sobelKer('x')
                sobelx = signal.convolve2d(self.__imgCache[0], sobelKernel,  boundary='symm', mode='same')
                self.__imgCache[1] = sobelx
                sobelx = np.abs(sobelx)
                smin = sobelx.min()
                smax = sobelx.max()
                diff = smax - smin
                sobelx = ((sobelx - smin) / diff) * 255
                sobelx = sobelx.astype(np.uint8)
                cv2.imshow('original', self.__imgCache[0])
                cv2.imshow('Sobel X', sobelx)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("3.1 should be done first!")
                self.__msg.exec_()

        elif bid == '3.3' :
            if self.legal[2] >= 2:
                self.legal[2] = 4 if self.legal[2] == 3 else 3
                sobelKernel = self.__sobelKer('y')
                sobely = signal.convolve2d(self.__imgCache[0], sobelKernel,  boundary='symm', mode='same')
                self.__imgCache[2] = sobely
                sobely = np.abs(sobely)
                smin = sobely.min()
                smax = sobely.max()
                diff = smax - smin
                sobely = ((sobely - smin) / diff) * 255
                sobely = sobely.astype(np.uint8)
                cv2.imshow('original', self.__imgCache[0])
                cv2.imshow('Sobel Y', sobely)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("3.1 should be done first!")
                self.__msg.exec_()

        elif bid == '3.4' :
            if self.legal[2] >= 4:
                magnitude = np.sqrt(self.__imgCache[1] ** 2 + self.__imgCache[2] ** 2)
                magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())) * 255
                magnitude = magnitude.astype(np.uint8)
                cv2.imshow('Magnitude', magnitude)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("3.2 3.3 should be done first!")
                self.__msg.exec_()

        elif bid == '4.1' :
            if self.legal[3] >= 1:
                rotate = float(self.__rotateText.text())
                scale = float(self.__scaleText.text())
                tx = int(self.__TxText.text())
                ty = int(self.__TyText.text())

                self.__imgCenter = self.__imgCenter + np.array([tx, ty])
                h, w = self.pic.shape[:2]
                center = (self.__imgCenter[0], self.__imgCenter[1])
                M = np.float32([[1, 0, tx],[0, 1, ty]])
                res = cv2.warpAffine(self.pic, M, (w,h))
                M = self.__M(rotate, scale)
                res = cv2.warpAffine(res, M, (w, h))
                self.pic = res
                cv2.imshow('My Image', res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else: 
                self.__msg.setText("1.1 should be done first!")
                self.__msg.exec_()

        elif bid == '5.1':
            self.legal[4] = 1
            self.__model = VGG()
            self.__model.showTrainImg()
            self.__model.setOneHotEncode()
            self.__model.buildModel()

        elif bid == '5.2':
            if self.legal[4] >= 1:
                self.__model.printHyperparameter()
            else: 
                self.__msg.setText("5.1 should be done first!")
                self.__msg.exec_()

        elif bid == '5.3':
            if self.legal[4] >= 1:
                self.__model.printSummary()
            else: 
                self.__msg.setText("5.1 should be done first!")
                self.__msg.exec_()

        elif bid == '5.4':
            if self.legal[4] >= 1:
                self.__model.getAccLossPlot()
            else: 
                self.__msg.setText("5.1 should be done first!")
                self.__msg.exec_()

        elif bid == '5.5':
            if self.legal[4] >= 1:
                index = int(self.__testImgIndex.text())
                self.__model.showTestRes(index)
            else: 
                self.__msg.setText("5.1 should be done first!")
                self.__msg.exec_()

    def __M(self, rot, scale):
        center = (self.__imgCenter[0], self.__imgCenter[1])
        cosx = math.cos(rot * math.pi / 180)
        sinx = math.sin(rot * math.pi / 180)
        a = scale * cosx
        b = scale * sinx
        x = (1 - a) * center[0] - b * center[1]
        y = b * center[0] + (1 - a) * center[1]
        return np.float32([[a, b, x], [-b, a, y]])

    def __illegalMsg(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("This is a message box")
        msg.setInformativeText("This is additional information")
        msg.setWindowTitle("MessageBox demo")
        msg.setDetailedText("The details are as follows:")
        a = msg.show()
        
    def __blendShow(self, x):
        dst = cv2.addWeighted(self.__imgCache[0], x / 255, self.__imgCache[1], (255 - x) / 255, 0.0)
        cv2.imshow("blending", dst)

    def __gaussianKer(self, size, sigma = np.sqrt(0.5)):
        half = size // 2
        x, y = np.mgrid[-half : half + 1, -half : half + 1]
        gnorm = np.exp(-(x**2 + y**2)) 
        gnorm = gnorm / gnorm.sum()
        return gnorm

    def __sobelKer(self, dir):
        if dir == 'x':
            return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        if dir == 'y':
            return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return None

    def __sumMax(self, sum):
        index = 0
        m = 0
        for i in range(len(sum)):
            if sum[i] > m:
                m = sum[i]
                index = i
        return index, m

    def __resetImgCache(self):
        self.__imgCache = [0] * 8

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(app.exec_())