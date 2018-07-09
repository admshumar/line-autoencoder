
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import skimage.transform as transform
from PIL import Image

rotation = 1

#Image dimensions
width = 50
h_width = int(math.floor(width/2))
height = 50

#Line properties
border = 10 #Distance of a vertical lines' end to the top of the image
line_width = 1

#Layers
layer_1 = width*height
layer_2 = 500

#NOTE: batch size is determined by the number of rotations performed
#on the image of a vertical line.

#Input
def LineList():
    X = np.zeros((width, height), dtype=float)
    
    #Construct a vertical line whose length is "height" and whose
    #center is the "center" of the array.
    for i in range(border, height - border + 1):
        for j in range(h_width - line_width, h_width + line_width + 1):
            X[j,i] = 1.0

    samples = []

    for theta in range(1, 181, rotation):
        
        #Rotate our vertical line counterclockwise by angle "theta".
        Y = 255*interp.rotate(X, theta)
        
        #In general, interp.rotate outputs an array whose 
        #size differs from the input, so:
        if Y.shape != (width,height):
            Y = transform.resize(Y, (width,height))
        
        #Cast to torch.tensor.
        Y = torch.from_numpy(Y)
        
        #Reshape into a row vector.
        Y = torch.reshape(Y,(1,layer_1))
        
        #Cast to torch.floattensor. Otherwise torch.cat
        #interprets the elements of "samples" as torch.DoubleTensors.
        Y = Y.float()
        
        samples.append(Y)
        
    Z = torch.Tensor(len(samples), layer_1)
    Z = torch.cat(samples, out=Z)
    
    return Z

samples = LineList()


# In[2]:


#Image display functions.
def showImage(Z):
    return plt.show(plt.imshow(Z,cmap='gray'))

def showInputImage(sample):
    old_sample = torch.reshape(sample, (height, width))
    old_sample = old_sample.data.numpy()
    showImage(old_sample)

def showProcessedImage(sample, id_approx):
    shaped_sample = torch.reshape(sample,(height*width,1))
    processed_sample = torch.mm(id_approx, shaped_sample)
    processed_sample = torch.reshape(processed_sample, (height, width))
    processed_sample = processed_sample.data.numpy()
    showImage(processed_sample)


# In[3]:


class AutoEncoder(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.encoder = nn.Linear(layer_1, layer_2)
        self.decoder = nn.Linear(layer_2, layer_1)
        
    def forward(self, x):
        
        encoded_x = F.relu(self.encoder(x))
        decoded_x = F.relu(self.decoder(encoded_x))
        
        return decoded_x

model = AutoEncoder()


# In[4]:


#Loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.25)


# In[ ]:


#Training.
c=0
for sample in samples:
    for k in range (0,5):
        print("ITERATION:", c)
        c = c + 1
        
        params = list(model.parameters())
        g = params[0]
        f = params[2]
        id_approx = torch.mm(f,g)
        
        output = model(sample)
        showInputImage(sample)
        showProcessedImage(output, id_approx)
        
        loss = criterion(output, sample)
        loss.backward()
        optimizer.step()
        
        print("LOSS:", loss)
        for param in model.parameters():
            print(" PARAMETER:", param.size(), "\n", "PARAMETER NORM:", torch.norm(param))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

for param in model.parameters():
    print(param)


# In[ ]:


silent = 0.01*np.ones(( width*height, 1), dtype=float)
showImage(silent)
silent = torch.from_numpy(silent)
silent = silent.float()
silent = torch.mm(id_approx,silent)
silent = torch.reshape(silent, (width,height))
silent = silent.detach().numpy()
showImage(silent)

