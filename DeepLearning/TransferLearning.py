import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

def classifyImg(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # print(x)
    x = np.expand_dims(x, axis = 0)
    # print(x)
    x = preprocess_input(x)

    # Create the model
    model = ResNet152V2(weights='imagenet')

    # output prediction
    prediction = model.predict(x)

    # decode output
    print('prediction: ', decode_predictions(prediction, top=3)[0])

img_path1 = 'fig/monkey.jpg'
classifyImg(img_path1)
img_path2 = 'fig/bird.jpg'
classifyImg(img_path2)
img_path3 = 'fig/taj.jpg'
classifyImg(img_path3)
img_path4 = 'fig/car.jpeg'
classifyImg(img_path4)