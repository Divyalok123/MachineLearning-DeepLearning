import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt

# preparing the data
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255

# Converting to one-hot format
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

# A subroutine to display sample
def display_sample(samplenumber):
    # print(train_labels[samplenumber])
    label = train_labels[samplenumber].argmax(axis = 0)
    image = train_images[samplenumber].reshape([28, 28])
    plt.title('Sample: %d  Label: %d' % (samplenumber, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.savefig('fig/image_4.jpg')

# display_sample(723)

# creating the network
model = Sequential()

## 2D convolution set up to take 32 filters of each image and each filter being 3*3 size
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
## Second convolution to take 64 filters
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
## Pooling to distill the results into something more manageable
model.add(MaxPool2D(pool_size=(2,2)))
## Dropout to prevent overfitting
model.add(Dropout(0.25))
## Flattening for traditional multilayer perceptron
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=32, epochs=5, validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss: ', score[0])
print('Test Accuracy: ', score[1]) # 99.22% accuracy with just 5 epochs. Awesome!