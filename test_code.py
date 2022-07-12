# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from osgeo import gdal

### functions ---------------------------------------------------------

def list_files(catalog, wd = os.getcwd()):
    path = os.path.join(wd, catalog)
    files = os.listdir(path)
    files = [i for i in files if i.endswith((".tif"))]
    files = [os.path.join(wd, path, i) for i in files]
    return files

def as_tensor(datapoint):
    file = gdal.Open(datapoint, gdal.GA_ReadOnly)
    # this reads multilayers directly as (z, x, y) 
    arr = file.ReadAsArray()
    geotrans = file.GetGeoTransform()
    tensor = tf.convert_to_tensor(arr)
    file = None
    if any(tf.shape(tensor) != [128, 128]):
        # maybe it is better to fill the empty space with 0?
        tensor = tf.image.resize(tensor, [128, 128], method = "nearest")
    return tensor

def load_images(datapoint):
    ## datapoint.numpy().decode("utf-8")    
    ref = as_tensor(datapoint["ref"])
    var = as_tensor(datapoint["var"])
    var = var / tf.math.reduce_max(var) # local scaling (should be global)
    return ref, var

def cnn_model():
    model = models.Sequential()
    # eventually there will be more input layers here (z > 1)
    model.add(layers.Conv2D(32, (3, 3), activation = "relu",
                            input_shape = (128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    return model

### code --------------------------------------------------------------

# create paths to catalogs
paths = {"ref": list_files("reference"),
         "var": list_files("variable")}

# tf.data.Dataset.list_files()
ds = tf.data.Dataset.from_tensor_slices(paths)
# images = ds.map(load_images, num_parallel_calls = tf.data.AUTOTUNE)
images = ds.map(lambda x: tf.py_function(load_images, [x], [tf.string]),
                num_parallel_calls = tf.data.AUTOTUNE)

model = cnn_model()
model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              # this dataset is highly unbalanced
              metrics = ["accuracy"])
# implement testset later
history = model.fit(images, epochs = 10)
