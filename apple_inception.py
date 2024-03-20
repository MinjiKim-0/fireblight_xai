import tensorflow as tf
from tensorflow import keras
import io
import itertools
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

# tf.debugging.set_log_device_placement(True)

# # 텐서 생성
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)
# # Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0


##################### 

class_names = ['00normal apple', '03Anthracnose apple', '04Marssonia blotch apple', '05fireblight apple', '06Valsa canker apple', '07Alternaria leaf spot apple']

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

seed = 42
batch_size = 32
img_size = (299, 299)
tf.random.set_seed(seed)

# img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)
trainval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.) # , validation_split=0.15
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255., validation_split=0.5)

# tensorflow_gpu
# train_dir = "/fire_blight/Training/cropped/apple_origin" # /media/visbic/MGTEC/fireblight
train_dir = "/fire_blight/Training/apple_dataset/train"
test_dir = "/fire_blight/Training/apple_dataset/test" # /media/visbic/MGTEC/fireblight

# pnu ai computing
# train_dir = "/home/id202188552/fire_blight/cropped/Training/apple" 
# val_dir = "/home/id202188552/fire_blight/cropped/Validation/apple"


# train_generator = img_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=4000, class_mode="categorical", seed=seed)
# val_generator = img_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=800, class_mode="categorical", seed=seed)

# batch_size 설정하면 model.fit에서 batch_size 설정한 것과 같은 효과이다
train_generator = trainval_datagen.flow_from_directory(train_dir,
    target_size=img_size,
    batch_size=batch_size,
    seed=seed) # 20740 # 53837 # 57204 # subset="training" # 67295 # 60568 # 53837

# val_generator = trainval_datagen.flow_from_directory(train_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     seed=seed,
#     subset="validation") # 5182 # 13458 # 10091

val_generator = test_datagen.flow_from_directory(test_dir,
    target_size=img_size,
    batch_size=batch_size,
    seed=seed,
    subset="training") # 5182 # 13458 # 10091 # 1643

test_generator = test_datagen.flow_from_directory(test_dir,
    target_size=img_size,
    batch_size=batch_size,
    seed=seed,
    subset="validation") # 3282 # 3282 # 3282 # 1639

test_images, test_labels = test_generator.next()

# model = keras.applications.Xception(
#     include_top=True,
#     weights=None, #"imagenet"
#     input_tensor=None,
#     input_shape=(299, 299, 3),
#     pooling=None,
#     classes=6,
#     classifier_activation="softmax",
# )


model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None, # "imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax",
)

# model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001,
    epsilon=1e-07
), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy', tf.keras.metrics.Recall()])

data_path = '/fire_blight/apple_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelpath = data_path + '/{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1)

log_dir = "logs_apple/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

# # Clear out prior logging data.pip
# !rm -rf logs_apple/image

# logdir = "logs_apple/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Define the basic TensorBoard callback.
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  rounded_labels=np.argmax(test_labels, axis=1)
  cm = sklearn.metrics.confusion_matrix(rounded_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# history = model.fit(train_generator, epochs=50, verbose=1, validation_data=val_generator, batch_size=2) 

# history = 
model.fit(train_generator, epochs=300, verbose=1, validation_data=val_generator, callbacks=[tensorboard_callback, cm_callback, early_stopping_callback]) # , validation_data=val_generator

model.save('/fire_blight/apple_model/InceptionV3/1002-1814.hdf5')

#, callbacks=[checkpointer, tensorboard_callback, cm_callback]
#early_stopping_callback

#  , steps_per_epoch=10, validation_steps=4

# val_accuracy = history.history['val_accuracy']
# train_accuracy = history.history['accuracy']

# x_len = np.arange(len(train_accuracy))
# plt.plot(x_len, val_accuracy, label='val_accuracy')
# plt.plot(x_len, train_accuracy, label='train_accuracy')

# plt.legend(loc='best')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()