
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import cv2
import os
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
data = []
labels = []
# dataset Path  
imagePaths = sorted(list(paths.list_images(os.getcwd() + '\\Images')))
random.seed(42)
random.shuffle(imagePaths)
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # make the image array 1d
    image = cv2.resize(image, (48, 48)).flatten()
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
#####
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, random_state=42)
#####
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# 3 layers
model = Sequential()
model.add(Dense(1024, input_shape=(6912,), activation="tanh"))
model.add(Dense(512, activation="tanh"))
model.add(Dense(len(lb.classes_), activation="softmax"))
learningRate = 0.01
EPOCHS = 200
opt = SGD(lr=learningRate)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=EPOCHS, batch_size=32)
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("machine learning Project")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
imagePaths = sorted(list(paths.list_images(os.getcwd() + '\\plot')))
imagenameString = imagePaths[len(imagePaths)-1];


num = imagenameString.split('\\', 1)[1]
num = num.split('\\', 1)[1]
num = num.split('\\', 1)[1]
num = int(num.split('.', 1)[0])+1
#save plot with unique name
savePath = os.getcwd()+'\\plot\\'+str(num)
plt.savefig(savePath)	
# evaluate the network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))
#serializing network and label binarizer
savePath = os.getcwd()+"\\model\\"+str(num)
model.save(savePath)
lbPath = os.getcwd()+"\\label\\"+str(num);
f = open(lbPath, "wb")
f.write(pickle.dumps(lb))
f.close()


