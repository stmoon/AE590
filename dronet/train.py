import utils
import os
import tensorflow as tf
import cnn_models
from keras import optimizers
import logz
from common_flags import FLAGS
import log_utils
from keras.callbacks import ModelCheckpoint

OUTPUT_PATH = '/home/stmoon/Project/AE590/dronet/output/'
TRAIN_DATA_PATH = '/home/stmoon/Project/AE590/deeplab/out'
TEST_DATA_PATH = '/home/stmoon/Project/AE590/deeplab/test_data'

train_datagen = utils.DroneDataGenerator(rotation_range=0.2, rescale=1./255, width_shift_range=0.2, height_shift_range=0.2)

train_data_generator = train_datagen.flow_from_directory(TRAIN_DATA_PATH)
test_data_generator = train_datagen.flow_from_directory(TEST_DATA_PATH)


model = cnn_models.resnet8(700,500,1,1)

# Serialize model into json
json_model_path = os.path.join(OUTPUT_PATH, "model.json")
utils.modelToJson(model, json_model_path)

optimizer = optimizers.Adam(decay=1e-5)


model.alpha = tf.Variable(1, trainable=False, name='alpha', dtype=tf.float32)
#model.beta  = tf.Variable(0, trainable=False, name='beta', dtype=tf.float32)

# Initialize number of samples for hard-mining
model.k_mse = tf.Variable(32, trainable=False, name='k_mse', dtype=tf.int32)
#model.k_entropy = tf.Variable(32, trainable=False, name='k_entropy', dtype=tf.int32)

model.compile(loss=[utils.hard_mining_mse(model.k_mse)],
              #utils.hard_mining_entropy(model.k_entropy)],
              optimizer=optimizer, 
              loss_weights=[model.alpha])
              #loss_weights=[model.alpha, model.beta])

# Save model with the lowest validation loss
weights_path = os.path.join(OUTPUT_PATH, 'weights_{epoch:03d}.h5')
writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)
# Save model every 'log_rate' epochs.
# Save training and validation losses.
logz.configure_output_dir(OUTPUT_PATH)
saveModelAndLoss = log_utils.MyCallback(filepath=OUTPUT_PATH,
                         period=10,
                         batch_size=32)
 
model.fit_generator(train_data_generator, 
        steps_per_epoch= 15,
        epochs= 50, 
        callbacks=[writeBestModel, saveModelAndLoss],
        validation_data=test_data_generator,
        validation_steps=5,
        verbose=True)

