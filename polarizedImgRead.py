# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:58:55 2021

@author: Kert PC
"""

import os
import cv2 as cv
import numpy as np
from PIL import Image


path = 'F:/Diploma/dataset/'
dataSet = os.listdir(path)

dataSet 

for file in dataSet:
    fileName = file
    image = cv.imread(os.path.join(path, fileName), 0)
    
    h = len(image)
    w = len(image[0])
    d = 4
    
    if h % d == 0 and w % d == 0 :
        w0 = int(w / d)
        h0 = int(h / d)
        
        subImgs = []
        for j in range(0, d) : 
            subImgs.append(np.zeros((h0, w0)))
    
        for j in range(0, d) :
            if j == 0:
                x = np.array(list(range(1, w, 2)))
                y = np.array(list(range(1, h, 2)))
            if j == 1:
                x = np.array(list(range(0, w, 2)))
                y = np.array(list(range(0, h, 2)))
            if j == 2:
                x = np.array(list(range(0, w, 2)))
                y = np.array(list(range(1, h, 2))) 
            if j == 3:
                x = np.array(list(range(1, w, 2)))
                y = np.array(list(range(0, h, 2)))
                
                
            subImgs[j] = image[np.ix_(y,x)]
    else :
        print('Picture of invalid size')
        
    subIdx = 1  
    baseName = fileName.split('.')[0] + '_'
    polarFolder = 'F:/Diploma/dataset_split_half/'
    
    for subImg in subImgs :
        im = Image.fromarray(subImg)
        im = im.resize((608,512))
        im.save(polarFolder + baseName + ('00' + str(subIdx))[-2:] + '.bmp')
        subIdx = subIdx + 1
        
        
        