#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imread, imwrite

stack = []
for name, shift_y, shift_x in (
    ("ab", 36, 18),
    ("au", 35, 18),
    ("ib", 21, 17),
    ("iu", 20, 17),
    ):
    im = Image.open(name + ".png")
    im = np.array(im)
    im = im[shift_y:shift_y+366, shift_x:shift_x+248]
    print(im.shape)
    stack.append(im)
    imwrite(name + ".tif", im, imagej=True)
##    im = imread(name + ".tif")
##    print(im.shape, im.dtype)
stack = np.array(stack)
print(stack.shape)
imwrite('stack.tif', stack)

im = Image.open("analyte.png")
im = np.array(im)
im = np.clip(im, 0, 24)
im = np.expand_dims(im, 2)
im = np.tile(im, (1, 1, 4))
im = (im * (255 / im.max())).astype('uint8')
im[:, :, :3] = 255 - im[:, :, :3]
print(im.min(), im.max)
print(im.dtype)
imwrite("analyte.tif", im.astype('uint8'), imagej=True)
