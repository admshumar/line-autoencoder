
# coding: utf-8

# In[30]:


#Imports
import torch
import math
import numpy as np

import scipy.ndimage.interpolation as interp
import skimage.transform as transform
from PIL import Image

import imageio
import matplotlib.pyplot as plt

class Rectangle():
    
    def __init__(self, height, width, angle):
        self.height = height
        self.width = width
        self.angle = angle
        
class RectangleImage(Rectangle):
    
    def makeImage(height, width, border, line_height, line_width, theta):
        h_width = int(width/2)
        h_height = int(height/2)
        left_edge = h_width - int(line_width/2)
        right_edge = h_width + int(line_width/2)
        upper_edge = h_height - int(line_height/2)
        lower_edge = h_height + int(line_height/2)
        
        X = np.zeros((height, width), dtype=float)

        for j in range(upper_edge, lower_edge):
            for i in range(left_edge, right_edge):
                X[i,j] = 255.0

        #Rotate our vertical line counterclockwise by angle "theta".
        Y = interp.rotate(X, theta)

        #In general, interp.rotate outputs an array whose 
        #size differs from the input, so:
        if Y.shape != (height, width):
            Y = transform.resize(Y, (height, width))
    
        return Y
    
    def makeTorchRow(X):
        m = X.size
        d = torch.from_numpy(X)
        d = torch.reshape(d,(1, m))
        d = d.float()
        return d

    def makeTorchColumn(X):
        return torch.t(makeTorchRow(X))
    
    def __init__(self, height, width, border, line_height, line_width, angle):
        Rectangle.__init__(self, height, width, angle)
        self.line_height = line_height
        self.line_width = line_width
        self.border = border
        self.image = RectangleImage.makeImage(self.height, self.width, self.border, self.line_height, self.line_width, self.angle)
        self.TorchImage = torch.reshape(torch.from_numpy(self.image).float(), (1, 1, self.width, self.height))
        self.sample = RectangleImage.makeTorchRow(self.image)
        
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

