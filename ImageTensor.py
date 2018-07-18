#Imports
import os
import math
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

def makeTorchRow(X):
    m = X.size
    d = torch.from_numpy(X)
    d = torch.reshape(d,(1, m))
    d = d.float()
    return d

def makeTorchColumn(X):
    return torch.t(makeTorchRow(X))

def showImage(self):
    return plt.show(plt.imshow(self.image, cmap='gray'))

def writeImage(self):
    directory = "images/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + str(self.width) + "_" + str(self.height) + "_" + str(self.radius) + "_" + str(self.center_x) + "_" + str(self.center_y) +".jpeg"
    new_p = Image.fromarray(self.image)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save(filename, "JPEG")
