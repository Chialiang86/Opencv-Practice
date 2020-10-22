import sys, cv2
import numpy as np
import math

from scipy import signal
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit , QWidget, QGridLayout, \
    QGroupBox, QPushButton, QVBoxLayout, QFileDialog, QMessageBox
from ExtendedButton import ExtendedQPushButton

class MainWindow(QWidget):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.btnSet = []
        self.pic = None
        self.legal = [0, 0, 0, 0, 0]

        self.__imgCache = [0, 0, 0, 0, 0, 0, 0, 0]
        self.__imgCenter = np.array([160, 84])

        self.show()
        self.initUI() 
        self.setWindowTitle("2020 Opencvdl HW1")
        self.resize(600, 300)

    def initUI(self):

        #first
        self.loadImgBtn = ExtendedQPushButton('1.1 Load image')
        self.ColorSepBtn = ExtendedQPushButton('1.2 Color seperation')
        self.imgFlipBtn = ExtendedQPushButton('1.3 Image Flipping')
        self.BlendBtn = ExtendedQPushButton('1.4 Blending')

        #second
        self.midFilterBtn = ExtendedQPushButton('2.1 Medium Filter')
        self.gauBlur2Btn = ExtendedQPushButton('2.2 Gaussian Blur')
        self.bilateralBtn = ExtendedQPushButton('2.3 Bilateral')

        #third
        self.gauBlur3Btn = ExtendedQPushButton('3.1Gaussion Blur')
        self.sobelXBtn = ExtendedQPushButton('3.2 Sobel X')
        self.sobelYBtn = ExtendedQPushButton('3.3 Sobel Y')
        self.magnitudeBtn = ExtendedQPushButton('3.4 Magnitude')

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

        self.transformBtn = ExtendedQPushButton('4.Transformation')
        self.transformBtn.id = '4.1'
        self.transformBtn.clicked.connect(self.btnClickedHandler)


        grid = QGridLayout()
        gridTitle = ['1.Image Processing', '2.Image Smoothing', '3.Edge Detection', '4.Transformation']
        
        btnIds = [['1.1', '1.2', '1.3', '1.4'],
        ['2.1', '2.2', '2.3'],
        ['3.1', '3.2', '3.3', '3.4']]
        grid.addWidget(self.createThreeGroup([self.loadImgBtn, self.ColorSepBtn, self.imgFlipBtn, self.BlendBtn], \
            gridTitle[0], btnIds[0]), 0, 0)
        grid.addWidget(self.createThreeGroup([self.midFilterBtn, self.gauBlur2Btn, self.bilateralBtn], \
            gridTitle[1], btnIds[1]), 0, 1)
        grid.addWidget(self.createThreeGroup([self.gauBlur3Btn,  self.sobelXBtn, self.sobelYBtn, self.magnitudeBtn], \
            gridTitle[2], btnIds[2]), 0, 2)
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
            picFName, fileType = QFileDialog.getOpenFileName(self,"選取檔案","./","Picture file (*.JPG *.png *.jpg *.jpeg)")
            picFileType = picFName.split('.')[-1]
            if picFileType == 'jpg' or picFileType == 'png' or picFileType == 'jpeg':
                self.legal[0] = 1 # enable 1.2
                self.legal[1] = 1 # enable 2.1 - 2.3
                self.legal[2] = 1 # enable 3.1
                self.pic = cv2.imread(picFName)
                print('type:' + str(type(self.pic)))
                print('Height : ' + str(self.pic.shape[0]))
                print('Width : ' + str(self.pic.shape[1]))
                cv2.namedWindow('My Image',0)
                cv2.startWindowThread() 
                cv2.imshow('My Image', self.pic)
                k = cv2.waitKey(0)
                if k > 0:
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
            cv2.namedWindow('original',0)
            cv2.startWindowThread()
            cv2.namedWindow('Seperate',0)
            cv2.startWindowThread() 
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
            self.__imgCache[0] = resizedPic
            self.__imgCache[1] = resizedFlipping
            cv2.namedWindow('original',0)
            cv2.startWindowThread()
            cv2.namedWindow('flipping',0)
            cv2.startWindowThread() 
            cv2.imshow("original", resizedPic)
            cv2.imshow("flipping", resizedFlipping)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif bid == '1.4' and self.legal[0] > 1:
            cv2.namedWindow('blending')
            cv2.createTrackbar ( 'weight' , 'blending' , 0 , 255, self.__blendShow)
            cv2.setTrackbarPos ( 'weight' , 'blending' , 128)


        # blur
        elif bid == '2.1' and self.legal[1] > 0:
            midBlurImg = cv2.medianBlur(self.pic, 7) 
            cv2.namedWindow('original',0)
            cv2.startWindowThread()
            cv2.namedWindow('median',0)
            cv2.startWindowThread() 
            cv2.imshow('original', self.pic)
            cv2.imshow('median', midBlurImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif bid == '2.2' and self.legal[1] > 0:
            gauBlurImg = cv2.GaussianBlur(self.pic, (5,5), 0)
            cv2.namedWindow('original',0)
            cv2.startWindowThread()
            cv2.namedWindow('Gaussian',0)
            cv2.startWindowThread() 
            cv2.imshow('original', self.pic)
            cv2.imshow('Gaussian', gauBlurImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif bid == '2.3' and self.legal[1] > 0:
            bilBlurImg = cv2.bilateralFilter(self.pic, 9, 90, 90)
            cv2.namedWindow('original',0)
            cv2.startWindowThread()
            cv2.namedWindow('bilateral',0)
            cv2.startWindowThread() 
            cv2.imshow('original', self.pic)
            cv2.imshow('bilateral', bilBlurImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #edge detect
        elif bid == '3.1' and self.legal[2] >= 1:
            self.legal[2] = 2
            grayImg = cv2.cvtColor(self.pic, cv2.COLOR_BGR2GRAY)
            gauKernel = self.__gaussianKer(7)
            gauBlur = signal.convolve2d(grayImg, gauKernel,  boundary='symm', mode='same')
            gauBlur = gauBlur.astype(np.uint8)
            self.__imgCache[0] = gauBlur
            cv2.namedWindow('original',0)
            cv2.startWindowThread()
            cv2.namedWindow('Gaussian',0)
            cv2.startWindowThread() 
            cv2.imshow('original', grayImg)
            cv2.imshow('Gaussian', gauBlur)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif bid == '3.2' :
            if self.legal[2] >= 2:
                sobelKernel = self.__sobelKer('x')
                sobelx = signal.convolve2d(self.__imgCache[0], sobelKernel,  boundary='symm', mode='same')
                self.__imgCache[1] = sobelx
                sobelx = np.abs(sobelx)
                smin = sobelx.min()
                smax = sobelx.max()
                diff = smax - smin
                sobelx = ((sobelx - smin) / diff) * 255
                sobelx = sobelx.astype(np.uint8)
                cv2.namedWindow('original',0)
                cv2.startWindowThread()
                cv2.namedWindow('Sobel X',0)
                cv2.startWindowThread() 
                cv2.imshow('original', self.__imgCache[0])
                cv2.imshow('Sobel X', sobelx)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif bid == '3.3' :
            if self.legal[2] >= 2:
                sobelKernel = self.__sobelKer('y')
                sobely = signal.convolve2d(self.__imgCache[0], sobelKernel,  boundary='symm', mode='same')
                self.__imgCache[2] = sobely
                sobely = np.abs(sobely)
                smin = sobely.min()
                smax = sobely.max()
                diff = smax - smin
                sobely = ((sobely - smin) / diff) * 255
                sobely = sobely.astype(np.uint8)
                cv2.namedWindow('original',0)
                cv2.startWindowThread()
                cv2.namedWindow('Sobel Y',0)
                cv2.startWindowThread() 
                cv2.imshow('original', self.__imgCache[0])
                cv2.imshow('Sobel Y', sobely)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif bid == '3.4' :
            if self.legal[2] >= 2:
                magnitude = np.sqrt(self.__imgCache[1] ** 2 + self.__imgCache[2] ** 2)
                magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())) * 255
                magnitude = magnitude.astype(np.uint8)
                cv2.namedWindow('Magnitude',0)
                cv2.startWindowThread() 
                cv2.imshow('Magnitude', magnitude)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif bid == '4.1' :
            if self.legal[0] >= 1:
                rotate = float(self.rotateText.text())
                scale = float(self.scaleText.text())
                tx = int(self.TxText.text())
                ty = int(self.TyText.text())
                
                print('size:' + str(self.pic.shape))

                self.__imgCenter = self.__imgCenter + np.array([tx, ty])
                h, w = self.pic.shape[:2]
                center = (self.__imgCenter[0], self.__imgCenter[1])
                
                M = np.float32([[1, 0, tx],[0, 1, ty]])
                res = cv2.warpAffine(self.pic, M, (w,h))
                
                M = self.__M(rotate, scale)
                #M = cv2.getRotationMatrix2D(center, rotate, scale) 
                res = cv2.warpAffine(res, M, (w, h))
                
                cv2.imshow('My Image', res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



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
        #gnit = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2))
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



    def __conv2D(self, img, ker):
        dst = np.zeros(img.shape, dtype=img.dtype)
        bh = ker.shape[0] // 2
        bw = ker.shape[1] // 2
        ih = img.shape[0]
        iw = img.shape[1]
        kerSize = ker.shape[0] * ker.shape[1]
        for i in range(bw, iw - bw):
            for j in range(bh, ih - bh):
                submatrix = img[i-bw:i+bw+1,j-bh:j+bh+1]
                dst[i][j] = np.sum(submatrix * ker) / kerSize
                print(str(i) + ' ' + str(j) + ' ' + str(sum))
        dst[:bw,:] = img[:bw,:]
        dst[iw-bw:,:] = img[iw-bw:,:]
        dst[:,:bh] = img[:,:bh]
        dst[:,ih-bh:] = img[:,ih-bh:]
        return dst


    def __resetImgCache(self):
        self.__imgCache = [0, 0, 0, 0, 0, 0, 0, 0]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    sys.exit(app.exec_())