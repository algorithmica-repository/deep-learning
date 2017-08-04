from skimage import data
from skimage import io
from skimage import color
import os
import pandas as pd

coffee = data.coffee()
type(coffee)
coffee.shape
io.imshow(coffee)

coffee_gray = color.rgb2gray(coffee)
type(coffee_gray)
coffee_gray.shape
io.imshow(coffee_gray)

tmp1 = coffee_gray.reshape((1,-1))
type(tmp1)
tmp1.shape

tmp2 = tmp1.reshape((400,600))
tmp2.shape
io.imshow(tmp2)

os.getcwd()
os.chdir("D:\\digit_recognizer")

digit_train = pd.read_csv("train.csv")
digit_train.shape

image = digit_train.iloc[0,1:]
image.shape
image_orig = image.reshape([28,28])/255.0
image_orig.shape
io.imshow(image_orig)

