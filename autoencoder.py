
# coding: utf-8

# In[1]:


#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np
import scipy.ndimage.interpolation as interp

import skimage.transform as transform
from PIL import Image

rotation = 5

#Image dimensions
width = 400
h_width = int(math.floor(width/2))
height = 400

#Line properties
border = 20 #Distance of a vertical lines' end to the top of the image
line_width = 2

#Layers
layer_1 = width*height
layer_2 = 1

#NOTE: batch size is determined by the number of rotations performed
#on the image of a vertical line.

#Input
def LineList():
    X = np.zeros((width, height), dtype=float)
    
    #Construct a vertical line whose length is "height" and whose
    #center is the "center" of the array.
    for i in range(border, height - border + 1):
        for j in range(h_width - line_width, h_width + line_width + 1):
            X[i,j] = 1.0

    samples = []

    for theta in range(0, 180, rotation):
        
        #Rotate our vertical line counterclockwise by angle "theta".
        Y = 255*interp.rotate(X, theta)
        
        #In general, interp.rotate outputs an array whose 
        #size differs from the input, so:
        if Y.shape != (width,height):
            Y = transform.resize(Y, (width,height))
        
        #Cast to torch.tensor.
        Y = torch.from_numpy(Y)
        
        #Reshape for input into the autoencoder.
        Y = torch.reshape(Y,(1, layer_1))
        
        #Cast to torch.floattensor. Otherwise torch.cat
        #interprets the elements of "samples" as torch.DoubleTensors.
        Y = Y.float()
        
        samples.append(Y)
        
    Z = torch.Tensor(len(samples), layer_1)
    Z = torch.cat(samples, out=Z)
    
    return Z

samples = LineList()

#AutoEncoder class. (Thanks for your help, Daniel!)
class AutoEncoder(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.encoder = nn.Linear(layer_1, layer_2)
        self.decoder = nn.Linear(layer_2, layer_1)
        
    def forward(self, x):
        
        encoded_x = F.relu(self.encoder(x))
        decoded_x = F.relu(self.decoder(encoded_x))
        
        return decoded_x

#AutoEncoder instance and optimizer.
model = AutoEncoder()
optimizer = optim.SGD(model.parameters(), lr=0.05)

#Loss
criterion = nn.MSELoss()

for sample in samples:
    #sample = torch.reshape(sample, (layer_1,))
    output = model(sample)
    loss = criterion(output, sample)
    loss.backward()
    optimizer.step()
    print("LOSS:", loss)
    for param in model.parameters():
        print(" PARAMETER:", param.size(), "\n", "PARAMETER NORM:", torch.norm(param))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

for param in model.parameters():
    print(param)

