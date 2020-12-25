# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:40:00 2020

@author: Abdullah
"""
import skimage.io as io
from skimage import data_dir
from skimage.transform import resize
from skimage.feature import hog
import time
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


start = time.time()
cats_imgs = io.ImageCollection('train/cat*.jpg')
dogs_imgs = io.ImageCollection('train/dog*.jpg')
testData = []
testlabels=[]
print(data_dir)

print(len(cats_imgs))
cats = []
dogs = []
trainLabels=[]
#cats_temp = [hog(resize(img,(128,64))  , orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True) for img in cats_imgs]

for index, img in zip(range(1000), cats_imgs):
    recized_img =  resize(img,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    cats.append(fd)
    trainLabels.append(1)
#print(len(cats_temp))
for index, img in zip(range(1000,1500), cats_imgs):
    recized_img =  resize(img,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    testData.append(fd)
    testlabels.append(1)

for index, img in zip(range(1000,1500), dogs_imgs):
    recized_img =  resize(img,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    testData.append(fd)
    testlabels.append(0)


#dogs_temp = [ hog(resize(img,(128,64))  , orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True) for img in dogs_imgs]


for index, img in zip(range(1000),dogs_imgs):
    recized_img =  resize(img,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    dogs.append(fd)
    trainLabels.append(0)

#print(len(dogs))
df_cats = pd.DataFrame(cats)
df_dogs = pd.DataFrame(dogs)
testData = pd.DataFrame(testData)
print ()
trainData= [ df_cats, df_dogs]
trainData = pd.concat(trainData)

model = LinearSVC()
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
print(classification_report(testlabels, predictions,))
