import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from torchvision.models import DenseNet

from VGG16 import VGG16
from hisModel import hisModel
from oldModel import  oldModel
from diff_VGG import diff_VGG

images = glob.glob('cars/*.jpg')

data = pd.DataFrame(images, columns=['src'])

data['brand'] = data['src'].apply(lambda x: x.split('_')[0].split('\\')[-1])
print(data['brand'].value_counts().head(10).to_string())

X = []
y = []

for i in range(len(data)):
    src = data.loc[i, 'src']
    src = cv2.imread(src, cv2.IMREAD_COLOR)
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    X.append(cv2.resize(dst, dsize=(224, 224)))
    y.append(data.loc[i, 'brand'])

car_brands = ["Ford", "Kia", "BMW"]

for i in range(len(y)):
    if y[i] == car_brands[0]:
        y[i] = 0
    elif y[i] == car_brands[1]:
        y[i] = 1
    else:
        y[i] = 2
y = to_categorical(y, num_classes=3)

X = np.array(X)
X = X.astype('float32')
X = X / 255.0
X = X.reshape(-1, 224, 224, 1)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)

EPOCHS = 10
BATCH_SIZE =15

datagen = ImageDataGenerator(rotation_range=0.5,
                             zoom_range=0.5,
                             width_shift_range=0.5,
                             height_shift_range=0.5)
datagen.fit(X_train)

model = oldModel()
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // BATCH_SIZE)

model.evaluate(X_test, y_test)

plt.plot(history.history["val_accuracy"],color="r",label="val_accuracy")
plt.plot(history.history["accuracy"],color="b",label="train_accuracy")
plt.title("Accuracy Graph")
plt.xlabel("number of epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history["val_loss"],color="g",label="val_loss")
plt.plot(history.history["loss"],color="y",label="train_loss")
plt.title("Loss Graph")
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

img_gray = cv2.imread("C:\\Users\\jiami\\PycharmProjects\\architwankhade\\bmw_img.jpg", 0)
pred_img = np.array(img_gray)


plt.imshow(pred_img)
plt.axis('off')
plt.show()

pred_img.resize((224, 224, 1))

images_list = []
images_list.append(np.array(pred_img))

x = np.asarray(images_list)


pred = model.predict(x)
print(car_brands[(pred.argmax())])