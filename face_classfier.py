import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from keras.callbacks import EarlyStopping
import  pandas as pd
import  keras
from  sklearn import metrics
from  keras.applications.resnet50 import  ResNet50
from torchvision import  models,transforms
from keras.preprocessing import image
from sklearn.model_selection import  train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def DataSet():
    train_path_angry = './data/train_data/angry/'
    train_path_disgust= './data/train_data/disgust/'
    train_path_fear  = './data/train_data/fear/'
    train_path_happy = './data/train_data/happy/'
    train_path_surprise = './data/train_data/surprise/'

    test_path_angry = './data/test_data/angry/'
    test_path_disgust = './data/test_data/disgust/'
    test_path_fear  = './data/test_data/fear/'
    test_path_happy = './data/test_data/happy/'
    test_path_surprise = './data/test_data/surprise/'

    imglist_train_angry = os.listdir(train_path_angry)
    imglist_train_disgust = os.listdir(train_path_disgust)
    imglist_train_fear = os.listdir(train_path_fear)
    imglist_train_happy = os.listdir(train_path_happy)
    imglist_train_surprise = os.listdir(train_path_surprise)

    imglist_test_angry = os.listdir(test_path_angry)
    imglist_test_disgust = os.listdir(test_path_disgust)
    imglist_test_fear  = os.listdir(test_path_fear)
    imglist_test_happy = os.listdir(test_path_happy)
    imglist_test_surprise = os.listdir(test_path_surprise)


    #
    X_train = np.empty((len(imglist_train_disgust) + len(imglist_train_fear)+len(imglist_train_happy)+\
                        +len(imglist_train_angry)+len(imglist_train_surprise), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_disgust) + len(imglist_train_fear)+len(imglist_train_happy)\
                        +len(imglist_train_angry)+len(imglist_train_surprise), 5))

    count = 0
    #将 disgust 标记为1  并更改图片大小
    for img_name in imglist_train_disgust:
        img_path =train_path_disgust + img_name
        try:
            img = Image.open(img_path)
            img = img.resize((224,224),Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            print(img_name)
            print(img.shape)
            print(img)
            X_train[count] = img
            Y_train[count] = np.array((1, 0, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)


    #将fear 标记为2
    for img_name in imglist_train_fear:
        try:
            img_path = train_path_fear + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 1, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)

    #happy 标记为3
    for img_name in imglist_train_happy:
        try:
            img_path = train_path_happy + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 0, 1, 0, 0))
            count += 1
        except Exception as e:
            print(e)


    #将surprise 标记为4
    for img_name in imglist_train_surprise:
        try:
            img_path = train_path_surprise + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 0, 0, 1, 0))
            count += 1
        except Exception as e:
            print(e)

    # 将angry 标记为5
    for img_name in imglist_train_angry:
        try:
            img_path = train_path_angry + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 0, 0, 0, 1))
            count += 1
        except Exception as e:
            print(e)

    X_train = X_train[0:count]
    Y_train = Y_train[0:count]
    print(count)
    print(X_train.shape)
    print(Y_train.shape)

    #对于test集合的处理
    X_test = np.empty((len(imglist_test_disgust) + len(imglist_test_fear)+len(imglist_test_happy)+ \
                           len(imglist_test_angry)+len(imglist_test_surprise), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_disgust) + len(imglist_test_fear)+len(imglist_test_happy)+ \
                           len(imglist_test_angry)+len(imglist_test_surprise), 5))
    count = 0

    # 将 disgust 标记为1
    for img_name in imglist_test_disgust:
        try:
            img_path = test_path_disgust + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_test[count] = img
            Y_test[count] = np.array((1, 0, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)

    # 将fear 标记为2
    for img_name in imglist_test_fear:
        try:
            img_path = test_path_fear + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_test[count] = img
            Y_test[count] = np.array((0, 1, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)

    # happy 标记为3
    for img_name in imglist_test_happy:


        try:
            img_path = test_path_happy + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            Y_test[count] = np.array((0, 0, 1, 0, 0))
            count += 1
            X_test[count] = img
        except Exception as e:
            print(e)

    # 将surprise 标记为4
    for img_name in imglist_test_surprise:
        try:
            img_path = test_path_surprise + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            Y_test[count] = np.array((0, 0, 0, 1, 0))
            count += 1

            X_test[count] = img
        except Exception as e:
            print(e)

    # 将angry 标记为5
    for img_name in imglist_test_angry:

        try:
            img_path = test_path_angry + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            Y_test[count] = np.array((0, 0, 0, 0, 1))
            count += 1
            X_test[count] = img
        except Exception as e:
            print(e)
    print(count)
    X_test = X_test[0:count]
    Y_test = Y_test[0:count]
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]


    # index = [i for i in range(len(X_train))]
    # random.shuffle(index)
    # X_train = X_train[index]
    # Y_train = Y_train[index]
    #
    # index = [i for i in range(len(X_test))]
    # random.shuffle(index)
    # X_test = X_test[index]
    # Y_test = Y_test[index]

    return X_train, Y_train, X_test, Y_test
def add_new_last_layer(base_model,nb_classes):
    x = base_model.output
    x2 =keras.layers.Dense(nb_classes,activation='softmax')(x)
    model = keras.Model(input = base_model.input,output = x2)
    return model


if __name__ =="__main__":

    #disgust:1   fear:2 happy:3 surprise:4 angry:5
    x_train, y_train, x_test, y_test = DataSet()
    print('X_train shape : ', x_train.shape)
    print('Y_train shape : ', y_train.shape)
    print('X_test shape : ', x_test.shape)
    print('Y_test shape : ', y_test.shape)

    # # model

    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.1,random_state=0)

    base_model = ResNet50(weights='imagenet')
    model = add_new_last_layer(base_model=base_model,nb_classes=5)

    print(model.summary())
    model.summary()
    model.layers.pop()
    print(model.summary)
    # # train
    early_stopping = EarlyStopping(monitor='val_acc')
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=20, batch_size=32,validation_split=0.2,shuffle=True)

    # # evaluate
    y_test = np.array(y_test)
    print(y_test)
    y_true = np.argmax(y_test,axis=1)
    y_pred = model.predict(x_test)
    y_pred = np.array(y_pred)
    print(y_pred)
    print(y_true)
    y_pred = np.argmax(y_pred,axis=1)
    print(y_pred)
    # pre_csv  = pd.DataFrame(y_pred)
    ac = metrics.accuracy_score(y_true ,y_pred)
    print("accuracy_score   %f" % ac)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    print("f1_score  %f" % f1)
    target_names = ['disgust','fear', 'happy','surprise','angry']
    print(metrics.classification_report(y_true, y_pred,target_names=target_names))
    # pre_csv.to_csv("1.txt",index =False)


    # # save

    #
    # model.save('my_resnet_model.h5')
    # #
    # # # # restore
    # #
    # #
    # model = keras.models.load_model('my_resnet_model.h5')
    #
    # # # test
    #
    #
    img_path = "./data/happy_show.jpg"
    img_path2 = "./data/fear_show.jpg"
    img_list = [img_path,img_path2]
    for img_path in img_list:
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ADAPTIVE)
        #
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # 为batch添加第四维
        print(img_path)
        print(model.predict(img))
        result =np.argmax(model.predict(img))
        if result ==0:
            print("disgust")
        elif result ==1:
            print("fear")
        elif result==2:
            print("happy")
        elif result ==3:
            print("surprise")
        elif result==4:
            print("angry")

