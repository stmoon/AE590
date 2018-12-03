from keras.models import model_from_json
import numpy
import os
import utils

MODEL_PATH = '/home/stmoon/Project/AE590/dronet/output/model.json'
WEIGHTS_PATH = '/home/stmoon/Project/AE590/dronet/output/weights_010.h5'
TEST_DATA_PATH = '/home/stmoon/Project/AE590/deeplab/test_data'

# test data
datagen = utils.DroneDataGenerator(rescale=1./255)
test_data_generator = datagen.flow_from_directory(TEST_DATA_PATH)
X, y = test_data_generator.next()
print(y.shape)

# load json and create model
json_file = open(MODEL_PATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(WEIGHTS_PATH)
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mse', optimizer='adam')
score = loaded_model.evaluate(X, y, verbose=0)
print(score)

