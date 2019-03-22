
from keras.models import load_model
import argparse
import pickle
import cv2

imagePath = input("Image Path");
ModelPath = input("Model Path");
lablePath = input("lablePath");
image = cv2.imread(imagePath)
output = image.copy()
image = cv2.resize(image, (48, 48))
image = image.astype("float") / 255.0
image = image.flatten()
image = image.reshape((1, image.shape[0]))
model = load_model(ModelPath)
lb = pickle.loads(open(lablePath, "rb").read())
 
# make a prediction on the image
preds = model.predict(image)
 
# find the class label index with the largest corresponding
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)
 
# show the output image
cv2.imshow("Image", output)
print(text)
cv2.waitKey(0)
