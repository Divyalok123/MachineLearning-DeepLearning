import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

# preparing data
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# one hot encoding
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

# function to display samples
def display_sample(num):
    # print(train_labels[num])  

    #Print the label converted back to a number
    label = train_labels[num].argmax(axis=0)

    #Reshape back to 28*28 image
    image = train_images[num].reshape([28,28])
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.savefig('fig/image_3.jpg')
    
# display_sample(784)

# creating the network
model = Sequential() 
model.add(Dense(632, activation='relu', input_shape=(784,))) 
model.add(Dropout(0.3)) 
model.add(Dense(632, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(10, activation='softmax'))

# model description
# model.summary()

# setting up optimizer, loss functions and metrics we need
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy']) 

# training the model
history = model.fit(train_images, train_labels,
                    batch_size=350,
                    epochs=30,
                    verbose=2,
                    validation_data=(test_images, test_labels))

# evaluating our model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0]) #0.1213
print('Test accuracy:', score[1]) # 98.43%

# visualizing the wrong results
import matplotlib.backends.backend_pdf as bpdf
pdfpage = bpdf.PdfPages('fig/falseoutputs_K.pdf')

with pdfpage as pdf:
    for x in range(1000):
        test_image = test_images[x].reshape(1,784)
        predicted_category = model.predict(test_image).argmax()
        label = test_labels[x].argmax()
        if (predicted_category != label):
            plt.figure()
            plt.title('Label: %d, Prediction: %d ' % (label, predicted_category))
            img = plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r')).get_figure()
            pdf.savefig(img)
