from skimage import data
from skimage import io
from skimage import color
from skimage import filters
import numpy as np

coffee = data.coffee()
io.imshow(coffee)

#convert color image to gray image
coffee_gray = color.rgb2gray(coffee)
io.imshow(coffee_gray)

#compute gradients of image along y-axis
grad_y = np.diff(coffee_gray, axis=0)
io.imshow(grad_y)

#compute gradients of image along x-axis
grad_x = np.diff(coffee_gray, axis=1)
io.imshow(grad_x)

#working with filters

#guassian filter
filtered1 = filters.gaussian(coffee, sigma=3)
io.imshow(filtered1)
filtered2 = filters.gaussian(coffee, sigma=[1,10])
io.imshow(filtered2)

#sobel filter for edge detection
text = data.text()
io.imshow(text)
edges = filters.sobel(text)
io.imshow(edges)

#canny edge detector
edges1 = filters.canny(coffee_gray)
io.imshow(edges1)
edges2 = filters.canny(coffee_gray, sigma=2)
io.imshow(edges2)
edges3 = filters.canny(coffee_gray, sigma=2, low_threshold=0.2, high_threshold=0.5)
io.imshow(edges3)

#feature detection
from skimage.feature import corner_harris, peak_local_max, corner_peaks
