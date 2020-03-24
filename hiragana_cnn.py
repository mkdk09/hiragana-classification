import glob
import os
import re
import keras
from keras import callbacks
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

# データ生成は時間かかるので別ファイルで生成してnpyデータを保存しておく
X = []
Y = []
for directory in glob.glob("hiragana73/*"):
    moji = directory.split('/')[1]
    b = 'b\'\\\\' + moji.lower() + '\''
    count = 0
    for i in range(12353, 12436):
        if i==12353 or i==12355 or i == 12357 or i == 12359 or i == 12361 \
        or i == 12387 or i==12401 or i==12404 or i==12407 or i == 12410 or i==12413 \
        or i==12419 or i== 12421 or i==12423 or i == 12430:
            continue
            
        if str(chr(i).encode('unicode-escape'))== b:
            for picture in list_pictures(directory + '/'):
                img = img_to_array(
                    load_img(picture, target_size=(28, 28), grayscale=True))
                X.append(img)
                Y.append(count)
            # break
        count += 1

X = np.asarray(X)
Y = np.asarray(Y)
X = X.astype('float32')
X = X / 255.0
Y = np_utils.to_categorical(Y, 68)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2)

datagen.fit(X_train)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

test_datagen.fit(X_test)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(68))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

"""
history = model.fit(X_train, y_train,  # 画像とラベルデータ
                    batch_size=8,
                    epochs=10,     # エポック数の指定
                    verbose=1,         # ログ出力の指定. 0だとログが出ない
                    validation_data=(X_test, y_test))
"""

fit_callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')]

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                                steps_per_epoch=len(X_train) / 32, epochs=30,
                                verbose=1, validation_data=test_datagen.flow(X_test, y_test),
                                callbacks=fit_callbacks)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('./DA_hiragana_cnn_model.h5')
