import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
im = Image.open('stereo2012a.jpg')
fig, ax = plt.subplots()
ax.imshow(im)
uv = plt.ginput(6)