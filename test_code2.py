import os
import math
import skimage.io
import tensorflow as tf
from tensorflow.keras import layers, models


### functions ---------------------------------------------------------

def list_files(catalog, wd = os.getcwd()):
    path = os.path.join(wd, catalog)
    files = os.listdir(path)
    files = [i for i in files if i.endswith((".tif"))]
    files = [os.path.join(wd, path, i) for i in files]
    return files

def load_images(paths):
    lst = []
    for i in range(len(paths)):
        img = skimage.io.imread(paths[i])
        if img.shape != (128, 128):
            # order = 0 is nearest-neighbor resampling
            img = skimage.transform.resize(img, (128, 128), order = 0)
            # maybe it is better to fill the empty space with 0?
        lst.append(img)
    return lst
    
def preprocess_arrays(img_array, mask_array):
    
    img = tf.convert_to_tensor(img_array)
    img = tf.expand_dims(img, axis = 2)  
    # there will be separate transformation for each variable
    img = img / 2483 # this is maximum value of elevation
    
    mask = tf.convert_to_tensor(mask_array)
    mask = tf.expand_dims(mask_array, axis = 2)
    
    return img, mask

def cnn_model(num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = "relu",
                            input_shape = (128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = "relu"))
    model.add(layers.Dense(num_classes))
    return model


### code --------------------------------------------------------------

## load all images into memory
img_paths = list_files("variable")
smpl_imgs = load_images(img_paths)

mask_paths = list_files("reference")
smpl_masks = load_images(mask_paths)

##  create TF dataset
BATCHSIZE = 5
smpl_dataset = tf.data.Dataset.from_tensor_slices((smpl_imgs, smpl_masks))
smpl_dataset = smpl_dataset.map(preprocess_arrays)
smpl_dataset = smpl_dataset.shuffle(BATCHSIZE * 128, reshuffle_each_iteration = False)

# create training dataset
size = math.floor(len(smpl_imgs) * 0.8)
training_dataset = smpl_dataset.take(size)
training_dataset = training_dataset.batch(BATCHSIZE)

# create validation dataset
validation_dataset = smpl_dataset.skip(size)
validation_dataset = validation_dataset.batch(BATCHSIZE)

### are images in the training and test sets definitely not repeated?

### how to check distribution of training and validation datasets?
for images, labels in training_dataset.take(-1):
    x = labels.numpy()
x = x.reshape(-1)



## model training
model = cnn_model(num_classes = 11)
model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])
history = model.fit(training_dataset, validation_data = validation_dataset,
                    epochs = 10)
#> InvalidArgumentError:  logits and labels must have the same first dimension,
#> got logits shape [5,10] and labels shape [81920]