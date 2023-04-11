import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

(X_training, y_training), (X_test, y_test) = mnist.load_data()


# plt.imshow(X_training[3]) # With color
plt.imshow(X_training[5], cmap = 'gray')  # Without color
plt.title('Classe ' + str(y_training[5]))

predictors_training = X_training.reshape(X_training.shape[0], 28, 28, 1)
predictors_training = predictors_training.astype('float32')

predictors_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
predictors_test = predictors_test.astype('float32')

# Applying min-max normalization-
predictors_training /= 255
predictors_test /= 255

training_class = np_utils.to_categorical(y_training, 10)
