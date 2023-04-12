from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten,  Conv2D, MaxPooling2D
from keras.utils import np_utils
# augmentation step
from keras.preprocessing.image import ImageDataGenerator

(X_training, y_training), (X_test, y_test) = mnist.load_data()

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
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])

generator_training = ImageDataGenerator(rotation_range = 7,
                                        horizontal_flip = True,
                                        shear_range = 0.2,
                                        height_shift_range = 0.07,
                                        zoom_range = 0.2)
generator_test = ImageDataGenerator()

base_training = generator_training.flow(predictors_training, 
                                        training_class,
                                        batch_size = 128)
              
base_test = generator_test.flow(predictors_test, test_class)

classifier.fit_generator(base_training, steps_per_epoch = 60000 / 128,
                         epochs = 5, validation_data = base_test,
                         validation_steps = 10000 / 128)
                          
# result = classifier.evaluate(predictors_test, test_class)


