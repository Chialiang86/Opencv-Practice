import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def parseInt(l):
    for i in range(10):
        if l[i] == 1:
            return i


labelName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(trainImage, trainLabel), (testImage, testLabel) = cifar10.load_data()
model = keras.models.load_model('maxAcc.h5')

confusionMatrix = np.zeros((10, 10))

testImage = testImage[:100]

print('confusion matrix generating...')
score = model.predict(testImage, verbose=1)
predictList = []
for i in range(100):
    predictList.append(np.where(score[i] == max(score[i]))[0][0])

for i in range(100):
    confusionMatrix[predictList[i]][testLabel[i][0]] += 1

for i in range(10):
    for j in range(10):
        print(' ', confusionMatrix[i][j], end='')
    print('')


plt.bar(labelName, score[0], bottom=None, align='center')
plt.title('correct answer:' + labelName[0])
plt.xticks(rotation='vertical')
plt.show()




'''
h5 = glob.glob(r'.\*.h5')
(trainImage, trainLabel), (testImage, testLabel) = cifar10.load_data()
testLabel = keras.utils.to_categorical(testLabel, 10)

            
            rgbConcate = np.hstack((rsep, gsep, bsep))
            ratio1 = rgbConcate.shape[1] / 1200
            ratio2 = self.pic.shape[1] / 600
            rgbConcate = cv2.resize(rgbConcate, (int(rgbConcate.shape[1]/ratio1), int(rgbConcate.shape[0]/ratio1)))
            resizedPic = cv2.resize(self.pic, (int(self.pic.shape[1]/ratio2), int(self.pic.shape[0]/ratio2)))
            

for h in  h5:
    model = keras.models.load_model(h)
    score = model.evaluate(testImage, testLabel, verbose=0)
    print('-----------',h,'-------------')
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    '''

