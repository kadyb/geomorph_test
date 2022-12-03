import os
import numpy
import pandas
import skimage.io
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
exec(open("unet.py").read()) # load unet model


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
            # order == 0 is nearest-neighbor resampling
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

def plot_results(idx):
    true_img = smpl_imgs[idx]
    true_img = true_img[None, ..., None]
    pred_img = unet_model.predict(true_img)
    pred_img = tf.argmax(pred_img, axis = -1)
    pred_img = pred_img[0, :, :]
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.matshow(pred_img)
    ax1.set_title('PREDICTED')
    ax1.axis('off')
    ax2.matshow(smpl_masks[idx])
    ax2.set_title('TRUE')
    ax2.axis('off')

### code --------------------------------------------------------------

## load all images into memory
img_paths = list_files("variable")
smpl_imgs = load_images(img_paths)

mask_paths = list_files("reference")
smpl_masks = load_images(mask_paths)

## create TF dataset
BATCHSIZE = 5
smpl_dataset = tf.data.Dataset.from_tensor_slices((smpl_imgs, smpl_masks))
smpl_dataset = smpl_dataset.map(preprocess_arrays)
smpl_dataset = smpl_dataset.shuffle(BATCHSIZE * 128, reshuffle_each_iteration = False)

# create training dataset
size = numpy.floor(len(smpl_imgs) * 0.8)
training_dataset = smpl_dataset.take(size)
training_dataset = training_dataset.batch(BATCHSIZE)

# create validation dataset
validation_dataset = smpl_dataset.skip(size)
validation_dataset = validation_dataset.batch(BATCHSIZE)

### are images in the training and test sets definitely not repeated?

## check training data distribution
for images, labels in training_dataset.take(-1):
    distr = labels.numpy()
distr = distr.reshape(-1)
numpy.unique(distr)
numpy.max(distr) # maximum category ID
tab = pandas.DataFrame(distr).value_counts(normalize = True) * 100 # ID == 0 means NA


## model training
unet_model = build_unet_model(max_class = 26)
unet_model.compile(optimizer = tf.keras.optimizers.Adam(),
                   loss = "sparse_categorical_crossentropy",
                   metrics = "accuracy")
history = unet_model.fit(training_dataset, validation_data = validation_dataset,
                         epochs = 5)

## plot
pandas.DataFrame(history.history).plot(figsize = (8, 5))
plt.show()

## plot sample result
idx = numpy.random.choice(len(smpl_dataset), 1)[0]
plot_results(idx)

    
