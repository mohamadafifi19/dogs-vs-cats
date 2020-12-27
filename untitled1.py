# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:40:00 2020

"""
import skimage.io as io
from skimage import data_dir
from skimage.transform import resize
from skimage.feature import hog
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm, datasets

test_df = pd.read_csv('testSubmission.csv')
test_df = test_df[0:200]
print(len(test_df))


cats_imgs = io.ImageCollection('train/cat*.jpg')
dogs_imgs = io.ImageCollection('train/dog*.jpg')
#test_imgs = io.ImageCollection('test1/*.jpg')

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


for index, img in zip(range(1000),dogs_imgs):
    recized_img =  resize(img,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    dogs.append(fd)
    trainLabels.append(0)
    
'''
for index, img in zip(range(1000,1500), dogs_imgs):
    recized_img =  resize(img,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    testData.append(fd)
    testlabels.append(0)
    
'''
#print(len(dogs))
df_cats = pd.DataFrame(cats)
df_dogs = pd.DataFrame(dogs)
trainData= [ df_cats, df_dogs]
trainData = pd.concat(trainData)

model = LinearSVC(max_iter=5000)
model.fit(trainData, trainLabels)
#test
foldr='test1/'
form='.jpg'
test_feat=[]
real_label=[]
for i in range(1,201):
    test= io.imread(foldr+ str(i)+form)
    recized_img =  resize(test,(128,64))  
    fd, hog_img = hog(recized_img, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
    test_feat.append(fd)

prediction = model.predict(test_feat)
print(real_label)

real_label=test_df['label']
print (len(real_label))
print (len(prediction))
accuracy_score(real_label, prediction)




ris = datasets.load_iris()
X = test_feat  # we only take the first two features.
Y = real_label

# we create an instance of SVM and fit out data.
C = 0.1  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(trainData, trainLabels) #minimize hinge oss, One vs One
lin_svc = svm.LinearSVC(C=C).fit(trainData, trainLabels) #minimize squared hinge loss, One vs All
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(trainData, trainLabels)
poly_svc = svm.SVC(kernel='poly', degree=5, C=C).fit(trainData, trainLabels)


# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC',
          'SVC with RBF kernel',
          'SVC with polynomial kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == Y)
    print('accuracy :'+ str(titles[i]),accuracy)
   
