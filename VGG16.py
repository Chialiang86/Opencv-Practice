import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

size = (32, 32, 3)
class VGG():
    def __init__(self):
        self.__batch_size = 64
        self.__num_classes = 10
        self.__epochs = 20

        self.__model = Sequential()
        #self.__model = VGG16(weights='imagenet', include_top=False)
        (self.__trainImage, self.__trainLabel), (self.__testImage, self.__testLabel) = cifar10.load_data()
        self.__history = 0
        self.__classText = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    def loadData(self):
        (self.__trainImage, self.__trainLabel), (self.__testImage, self.__testLabel) = cifar10.load_data()
    
    def setOneHotEncode(self):
        self.__trainLabel = keras.utils.to_categorical(self.__trainLabel, self.__num_classes)
        self.__testLabel = keras.utils.to_categorical(self.__testLabel, self.__num_classes)

    def buildModel(self):
        self.__model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu', input_shape=(32,32,3)))
        self.__model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
        self.__model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        self.__model.add(Flatten())
        self.__model.add(Dense(4096, activation = 'relu'))
        self.__model.add(Dense(4096, activation = 'relu'))
        self.__model.add(Dense(10, activation = 'softmax'))
        self.__model.compile(loss='categorical_crossentropy', 
                  optimizer=RMSprop(lr=0.001), 
                  metrics=['accuracy'])
        return self.__model
    
    def train(self):
        self.__history = self.__model.fit(self.__trainImage, self.__trainLabel, batch_size=100, nb_epoch=20)
        return self.__history

    def printSummary(self):
        self.__model.summary()

    def printEvaluate(self):
        return self.__model.evaluate(self.__testImage, self.__testLabel, verbose=0)
        

    def showTrainImg(self):
        #32*32*3
        img10 = [0] * 10
        boolMap = [0] * 10
        for i in range(len(self.__trainLabel)):
            if boolMap[self.__trainLabel[i][0]] == 0:
                label = self.__trainLabel[i][0]
                boolMap[label] = 1
                img10[label] = self.__trainImage[i]
                if 0 not in boolMap:
                    break
        fig, ax = plt.subplots(2, 5, figsize=(8, 6))
        for i in range(10):
            ax[i//5, i%5].imshow(img10[i])
            ax[i//5, i%5].set_title(self.__classText[i])
        fig.tight_layout()
        plt.show()

model = VGG()
model.buildModel()
model.setOneHotEncode()

history = model.train()

res = model.printEvaluate()
print('Test loss:', res[0])
print('Test accuracy:', res[1])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')