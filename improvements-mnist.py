import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization.batch_normalization import BatchNormalization

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
test_class = np_utils.to_categorical(y_test, 10)

# Structure
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), 
                      activation = 'relu'))
# Improvements
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Flatten())

# Improvements
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten()) # Use just one time


classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])
classifier.fit(predictors_training, training_class,
               batch_size = 128, epochs = 5,
               validation_data = (predictors_test, test_class))

result = classifier.evaluate(predictors_test, test_class)