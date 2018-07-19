########
#IMPORTS
########

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import math
import numpy as np

import imageio
import matplotlib.pyplot as plt

import Rectangle

###########
#PARAMETERS
###########

#Rectangle parameters
scale = 5
width = 64*scale
height = 64*scale
border = 10
p = 1
line_height = 2
line_width = int(p*min(width,height))
max_side = int(min(width, height))
max_angle = 180
angle_list = np.arange(0, max_angle)
sample_number = int(len(angle_list))

#Neural network parameters
epoch = 50
layer_1 = width*height

########
#SAMPLES
########

#Randomly generate a list of rectangles with the same center
rectangleList = []
for theta in np.random.choice(angle_list, int(sample_number/2), replace=True):
    rect = Rectangle.RectangleImage(height, width, border, line_height, line_width, theta)
    rectangleList.append(rect)
    #d.writeImage()
    
#Convert a list of rectangles to a torch tensor
def makeTensor(rectangleList):
    height = rectangleList[0].height
    width = rectangleList[0].width
    
    W = torch.Tensor(len(rectangleList), 1, height, width)
    A = list(map(lambda rect:rect.TorchImage, rectangleList))
    
    W = torch.cat(A, out=W)
    return W

#Samples comprise a torch tensor derived from a list of disks.
samples = makeTensor(rectangleList)

############
#AUTOENCODER
############

class AutoEncoder(nn.Module):
    
    def Encode(i,j):
        maps = nn.Sequential(
                nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
                nn.MaxPool2d(2, padding=0),
                nn.LeakyReLU(0.2)
                )
        return maps

    def Decode(i,j):
        maps = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
                nn.LeakyReLU(0.2)
            )
        return maps
    
    def __init__(self):
        
        super().__init__()
        
        self.encode = nn.Sequential(
            AutoEncoder.Encode(1,8),
            AutoEncoder.Encode(8,4),
            AutoEncoder.Encode(4,4),
            AutoEncoder.Encode(4,3),
            #AutoEncoder.Encode(3,2),
            #AutoEncoder.Encode(2,1)
        )
        
        self.decode = nn.Sequential(
            #AutoEncoder.Decode(1,2),
            #AutoEncoder.Decode(2,3),
            AutoEncoder.Decode(3,4),
            AutoEncoder.Decode(4,4),
            AutoEncoder.Decode(4,8),
            AutoEncoder.Decode(8,1)
        )
        
    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

#AutoEncoder instance and optimizer.
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam's default learning rate is 0.001

c = 0
best_iteration = 0
best_loss = 10**10
for j in range (0, epoch):
    c = c + 1
    output = model(samples)
    loss = criterion(output, samples)
    loss.backward()
    optimizer.step()
    
    if best_loss > loss.detach().numpy():
        best_loss = loss.detach().numpy()
        best_iteration = c
    
    print("ITERATION:", c)
    print("LOSS:", loss.detach().numpy())
    #for param in model.parameters():
        #print(" PARAMETER:", param.size(), "\n", "PARAMETER NORM:", torch.norm(param))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

#for param in model.parameters():
    #print(param)

print("Best iteration:", best_iteration)
print("Best loss:", best_loss)

##############
#VISUALIZATION
##############

def getGrid(tensor, rows):
    if tensor.requires_grad == True:
        tensor = tensor.detach()
    #Make a visualizable grid of type torch.Tensor, convert to a numpy array, 
    #convert the dtype of the numpy array to 'uint8'.
    grid = torchvision.utils.make_grid(tensor, nrow=rows, padding = 100)
    grid = grid.permute(1,2,0)
    grid = grid.numpy()
    grid = grid.astype("uint8")
    return grid

# Plot an image using Matplotlib.
def plotSingleImage(tensor):
    if tensor.requires_grad == True:
        tensor = tensor.detach()
    # For multi-channel images, imshow needs a numpy array
    # with the channel dimension as the the last dimension.
    # For monochrome images, imshow needs only the spatial
    # dimensions.
    z = tensor.reshape(tensor.size()[1],tensor.size()[2])
    plt.imshow(z, cmap = "gray")
    plt.show()  
    
#Take a couple of sets of images and combine them in a grid for comparison.
def interlaceTorchTensors(x, y, col):
    w = torch.tensor([])
    a = int(x.shape[0])
    r = int(a/col)
    for i in range(0,r): 
        indices = torch.tensor(list(range(col*i, col*(i+1))))
        w_x = torch.index_select(x,0,indices)
        w_y = torch.index_select(y,0,indices)
        z = torch.cat((w_x,w_y),0)
        w = torch.cat((w,z),0)
    return w
    
#Write a grid to an image file.
def writeGrid(tensor, nrow, filename):
    grid = getGrid(tensor, nrow)
    plt.imshow(grid, cmap = "gray")
    imageio.imwrite(filename, grid)
    
#Display the inputs and outputs of the neural network.
def writeAutoEncoderAction(samples, col, filename):
    out = model(samples)
    if samples.requires_grad == True:
        samples = samples.detach()
    if out.requires_grad == True:
        out = out.detach()
    tensor = interlaceTorchTensors(samples, out, col)
    writeGrid(tensor, col, filename)
    
#writeAutoEncoderAction(samples, 10, "AutoEncoding.jpeg")
plotSingleImage(samples[4])

