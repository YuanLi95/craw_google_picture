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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def DataSet():
    train_path_bird = './data/train_data/bird/'
    train_path_cat = './data/train_data/cat/'
    train_path_dog  = './data/train_data/dog/'
    train_path_elephant = './data/train_data/elephant/'
    train_path_pandas = './data/train_data/pandas/'

    test_path_bird = './data/test_data/bird/'
    test_path_cat = './data/test_data/cat/'
    test_path_dog  = './data/test_data/dog/'
    test_path_elephant = './data/test_data/elephant/'
    test_path_pandas = './data/test_data/pandas/'

    imglist_train_bird = os.listdir(train_path_bird)
    imglist_train_cat = os.listdir(train_path_cat)
    imglist_train_dog = os.listdir(train_path_dog)
    imglist_train_elephant = os.listdir(train_path_elephant)
    imglist_train_pandas = os.listdir(train_path_pandas)

    imglist_test_bird = os.listdir(test_path_bird)
    imglist_test_cat = os.listdir(test_path_cat)
    imglist_test_dog  = os.listdir(test_path_dog)
    imglist_test_elephant = os.listdir(test_path_elephant)
    imglist_test_pandas = os.listdir(test_path_pandas)


    #
    X_train = np.empty((len(imglist_train_cat) + len(imglist_train_dog)+len(imglist_train_elephant)+\
                        +len(imglist_train_bird)+len(imglist_train_pandas), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_cat) + len(imglist_train_dog)+len(imglist_train_elephant)\
                        +len(imglist_train_bird)+len(imglist_train_pandas), 5))

    count = 0
    #将 cat 标记为1  并更改图片大小
    for img_name in imglist_train_cat:
        img_path =train_path_cat + img_name
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


    #将dog 标记为2
    for img_name in imglist_train_dog:
        try:
            img_path = train_path_dog + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 1, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)

    #elepant 标记为3
    for img_name in imglist_train_elephant:
        try:
            img_path = train_path_elephant + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 0, 1, 0, 0))
            count += 1
        except Exception as e:
            print(e)


    #将pandas 标记为4
    for img_name in imglist_train_pandas:
        try:
            img_path = train_path_pandas + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            Y_train[count] = np.array((0, 0, 0, 1, 0))
            count += 1
        except Exception as e:
            print(e)

    # 将bird 标记为5
    for img_name in imglist_train_bird:
        try:
            img_path = train_path_bird + img_name
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
    X_test = np.empty((len(imglist_test_cat) + len(imglist_test_dog)+len(imglist_test_elephant)+ \
                           len(imglist_test_bird)+len(imglist_test_pandas), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_cat) + len(imglist_test_dog)+len(imglist_test_elephant)+ \
                           len(imglist_test_bird)+len(imglist_test_pandas), 5))
    count = 0

    # 将 cat 标记为1
    for img_name in imglist_test_cat:
        try:
            img_path = test_path_cat + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_test[count] = img
            Y_test[count] = np.array((1, 0, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)

    # 将dog 标记为2
    for img_name in imglist_test_dog:
        try:
            img_path = test_path_dog + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            X_test[count] = img
            Y_test[count] = np.array((0, 1, 0, 0, 0))
            count += 1
        except Exception as e:
            print(e)

    # elepant 标记为3
    for img_name in imglist_test_elephant:


        try:
            img_path = test_path_elephant + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            Y_test[count] = np.array((0, 0, 1, 0, 0))
            count += 1
            X_test[count] = img
        except Exception as e:
            print(e)

    # 将pandas 标记为4
    for img_name in imglist_test_pandas:
        try:
            img_path = test_path_pandas + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.ADAPTIVE)
            img = image.img_to_array(img) / 255.0
            Y_test[count] = np.array((0, 0, 0, 1, 0))
            count += 1

            X_test[count] = img
        except Exception as e:
            print(e)

    # 将bird 标记为5
    for img_name in imglist_test_bird:

        try:
            img_path = test_path_bird + img_name
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

    #cat:1   dog:2 elephant:3 pandas:4 bird:5
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
    early_stopping = EarlyStopping(monitor='val_loss')
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
    target_names =['cat','dog' ,'elephant', 'pandas','bird']
    f1 = metrics.f1_score(y_true, y_pred,target_names=target_names)
    print("f1_score  %f" % f1)

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
    img_path = "./data/dog_show.jpg"
    img_path2 = "./data/cat_show.jpg"
    img_list = [img_path, img_path2]
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
            print("cat")
        elif result ==1:
            print("dog")
        elif result==2:
            print("elephant")
        elif result ==3:
            print("pandas")
        elif result==4:
            print("bird")

