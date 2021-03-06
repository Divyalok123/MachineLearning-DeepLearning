import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist #MNIST dataset
import matplotlib.pyplot as plt


num_classes = 10
num_features = 784

#preparing the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# print(x_train.shape)

x_train, x_test = x_train / 255., x_test / 255.


#function to show samples
def show_sample(num):
    label = y_train[num]
    image = x_train[num].reshape([28, 28])
    plt.title('Sample no. %d, Label %d' % (num, label))

    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.savefig('fig/image_1.jpg')

# show_sample(991)

# just to see what the neural network will working on
images = x_train[0].reshape([1, 784])

for i in range(1, 500):
    images = np.concatenate((images, x_train[i].reshape([1, 784])))

# plt.imshow(images, cmap=plt.get_cmap('gray_r'))
# plt.savefig('fig/image_2.jpg')

#hyper-parameters
learning_rate = 0.003
training_steps = 3200
batch_size = 350
display_step = 100

# Network parameters.
n_hidden = 632 # Number of neurons.

# shuffling and getting the initial data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)

#random value generator to initialize weights
random_normal = tf.initializers.RandomNormal()

weights = {
    'h': tf.Variable(random_normal([num_features, n_hidden])),
    'out': tf.Variable(random_normal([n_hidden, num_classes]))
}
biases = {
    'b': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

#model function
def neural_net(inputData):
    # Hidden fully connected layer with 512 neurons.
    hidden_layer = tf.add(tf.matmul(inputData, weights['h']), biases['b'])

    # Apply sigmoid to hidden_layer output for non-linearity.
    hidden_layer = tf.nn.sigmoid(hidden_layer)
    
    # Output fully connected layer with a neuron for each class.
    out_layer = tf.matmul(hidden_layer, weights['out']) + biases['out']

    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out_layer)

def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

optimizer = tf.keras.optimizers.SGD(learning_rate)

def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Training epoch: %i, Loss: %f, Accuracy: %f" % (step, loss, acc))

    # Test model on validation set.
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

n_images = 500
test_images = x_test[:n_images]
test_labels = y_test[:n_images]
predictions = neural_net(test_images)

import matplotlib.backends.backend_pdf as bpdf
pdfpage = bpdf.PdfPages('fig/falseoutputs.pdf')

with pdfpage as pdf:
    for i in range(n_images):
        model_prediction = np.argmax(predictions.numpy()[i])
        if (model_prediction != test_labels[i]):
            plt.figure()
            plt.title("Original Label: %i, Model Prediction: %i" % (test_labels[i], model_prediction))
            fig = plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray_r').get_figure()
            pdf.savefig(fig)

#final result -> 97.05% accuracy