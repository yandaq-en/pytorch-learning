from skimage import io
from skimage import transform
import os
import skimage


root = "../dataset/CATDOG/train/"
filenames = os.listdir(root)
for filename in filenames:
    image = io.imread(root+filename)
    image_resized = skimage.img_as_ubyte(transform.resize(image, (224, 224, 3)))
    io.imsave(root+filename, image_resized)
