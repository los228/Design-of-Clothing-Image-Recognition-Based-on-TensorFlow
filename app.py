from flask import Flask
import time
import os
import tensorflow as tf
import cv2
import numpy as np
# 导入flask库
from flask import Flask, render_template, request, jsonify
# 导入加载模型的方法load_model和模型编译使用的优化器类Adam
from keras.models import load_model
from keras.optimizers import Adam
# 导入pickle库
import pickle
app = Flask(__name__)


# 调用pickle库的load方法，加载图像数据处理时需要减去的像素均值pixel_mean
file = open('D:/2020大四上学期大学课程及其相关/1/resources\pixel_mean.pickle', 'rb')
pixel_mean = pickle.load(file)

model_filePath = 'D:/2020大四上学期大学课程及其相关/1/model.h5'
model = load_model(model_filePath)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

def classId_to_className(classId):
    category_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                     'Ankle boot']
    className = category_list[classId]
    return className




def get_imageNdarray(imageFilePath):
    image_ndarray = cv2.imread(imageFilePath)
    resized_image_ndarray = cv2.resize(image_ndarray, (32, 32), interpolation=cv2.INTER_AREA)
    return resized_image_ndarray

def process_imageNdarray(image_ndarray, pixel_mean):
    rgb_image_ndarray = image_ndarray[:, :, ::-1]
    image_ndarray_1 = rgb_image_ndarray / 255.0
    image_ndarray_2 = image_ndarray_1 - pixel_mean
    return image_ndarray_2

def predict_image(model, imageFilePath):
    image_ndarray = get_imageNdarray(imageFilePath)
    processed_image_ndarray = process_imageNdarray(image_ndarray, pixel_mean)
    inputs = processed_image_ndarray[np.newaxis, ...]
    probability_model = tf.keras.Sequential([
        model, tf.keras.layers.Softmax()
    ])
    predict_Y = probability_model.predict(inputs)[0]
    predict_y = np.argmax(predict_Y)
    predict_classId = predict_y
    predict_className = classId_to_className(predict_classId)
    print('对此图片路径 %s 的预测结果为 %s' %(imageFilePath, predict_className))
    return predict_className


# 使用predict_image这个API服务时的调用函数
@app.route("/predict_image", methods=['POST'])
def anyname_you_like():
    startTime = time.time()
    # 解析接收到的图片
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # 保存接收的图片到指定文件夹
        received_dirPath = 'D:/2020大四上学期大学课程及其相关/1/resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('接收图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        # 对指定图片路径的图片做分类预测，并打印耗时，返回预测种类名称
        startTime = time.time()
        predict_className = predict_image(model, imageFilePath)
        usedTime = time.time() - startTime
        print('完成对接收图片的分类预测，总共耗时%.2f秒\n' % usedTime)
        return jsonify(predict_className=predict_className)


@app.route('/')
def index_page():
    return render_template('homepage.html')


if __name__ == "__main__":
    print('在开启服务前，先测试predict_image函数')
    imageFilePath = 'D:/2020大四上学期大学课程及其相关/1/resources/001.jpg'
    predict_className = predict_image(model, imageFilePath)

    app.run("127.0.0.1", port=5000)
