from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.models import Model, Sequential, load_model
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import glob
import sys
from keras.preprocessing import image

test_data_dir = "./test/"

test_datagen = ImageDataGenerator(rescale=1. / 255)

img_width, img_height = 300, 300
nb_test_samples = 11
batch_size = 16
nb_category = 2

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

model = load_model('sneakers.hdf5')

# 画像を読み込んで予測する
def img_predict(filename):
    # 画像を読み込んで4次元テンソルへ変換
    img = image.load_img(filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
    # これを忘れると結果がおかしくなるので注意
    x = x / 255.0
    #表示
    #plt.imshow(img)
    #plt.show()
    # 指数表記を禁止にする
    np.set_printoptions(suppress=True)

    #画像の人物を予測
    pred = model.predict(x)[0]
    #結果を表示する
    print('[AF1, CONVERSE, OLDSKOOL, STANSMITH, YeezyBoost]')
    print(pred*100)

test = glob.glob('./test/*')

for t in test:
    img_predict(t)
