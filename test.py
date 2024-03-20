import tensorflow as tf
import sklearn.metrics
import numpy as np

seed = 42
batch_size = 3515 # 1639
img_size = (299, 299) # resnet은 224로
tf.random.set_seed(seed)

MODEL = "1006-1335.hdf5" # xception
# MODEL = "1002-1814.hdf5" # inceptionv3
# MODEL = "0927-1902.hdf5" # ResNet50V2

loaded_model = tf.keras.models.load_model(f'/fire_blight/apple_model/xception/{MODEL}') # xception
# loaded_model = tf.keras.models.load_model(f'/fire_blight/apple_model/InceptionV3/{MODEL}') # inceptionv3
# loaded_model = tf.keras.models.load_model(f'/fire_blight/apple_model/ResNet50V2/{MODEL}') # ResNet50V2

# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255., validation_split=0.5)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)

# test_dir = "/fire_blight/Training/apple_dataset/test" # /media/visbic/MGTEC/fireblight

test_dir = "/fire_blight/cyclegan"

test_generator = test_datagen.flow_from_directory(test_dir,
    target_size=img_size,
    batch_size=batch_size,
    seed=seed)#,
    # subset="validation") # 3282 # 3282 # 3282 # 1639

test_images, test_labels = test_generator.next()

# test_loss, test_acc = loaded_model.evaluate(test_images,  test_labels, verbose=2)

# print("test_images :", test_images.shape)

test_pred_raw = loaded_model.predict(test_images)
test_pred = np.argmax(test_pred_raw, axis=1)

rounded_labels=np.argmax(test_labels, axis=1)

# print("rounded_labels shape :", rounded_labels.shape)
# print("test_pred :", test_pred.shape)

# print("인덱스, 실제, 예측")
# for i in range(rounded_labels.shape[0]):
#     if rounded_labels[i] == test_pred[i] :
#         pass
#     else :
#         print(i, rounded_labels[i], test_pred[i])



accuracy = sklearn.metrics.accuracy_score(rounded_labels, test_pred)
recall = sklearn.metrics.recall_score(rounded_labels, test_pred, average=None)
cm = sklearn.metrics.confusion_matrix(rounded_labels, test_pred)


print("accuracy :", accuracy)
# print("recall :", recall[2])
print("recall :", recall)
print("cm :", cm)

