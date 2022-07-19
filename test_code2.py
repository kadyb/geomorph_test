import os
import tensorflow as tf
from skimage import io as skio


### functions ---------------------------------------------------------

def list_files(catalog, wd = os.getcwd()):
    path = os.path.join(wd, catalog)
    files = os.listdir(path)
    files = [i for i in files if i.endswith((".tif"))]
    files = [os.path.join(wd, path, i) for i in files]
    return files

def preprocess_arrays(img_array, mask_array):
    
    img = tf.convert_to_tensor(img_array)
    img = tf.expand_dims(img, axis = 2)
    if img.get_shape()[0:2] != [128, 128]:
        img = tf.image.resize(img, [128, 128])  
    img = img / tf.math.reduce_max(img)
    
    mask = tf.convert_to_tensor(mask_array)
    mask = tf.expand_dims(mask_array, axis = 2)
    if mask.get_shape()[0:2] != [128, 128]:
        mask = tf.image.resize(img, [128, 128])  
    
    return img, mask


### code --------------------------------------------------------------

img_paths = list_files("variable")
smpl_imgs = []
for i in range(len(img_paths)):
    smpl_imgs.append(skio.imread(img_paths[i]))

mask_paths = list_files("reference")
smpl_masks = []
for i in range(len(mask_paths)):
    smpl_masks.append(skio.imread(mask_paths[i]))


BATCHSIZE = 5

# create dataset from sample files
smpl_dataset = tf.data.Dataset.from_tensor_slices((smpl_imgs, smpl_masks))
#> ValueError: Can't convert non-rectangular Python sequence to Tensor.

smpl_dataset = smpl_dataset.map(preprocess_arrays) 
smpl_dataset = smpl_dataset.shuffle(BATCHSIZE * 128, reshuffle_each_iteration = False)
