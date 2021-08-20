# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:18:23 2021

@author: Kert PC
"""

import os
import cv2 as cv
import numpy as np

path = 'F:/Diploma/masks_renamed/'
fileSet = os.listdir(path)

i = 1

total = int(0)

types = []

types.insert(0, int(0))
types.insert(1, int(0))
types.insert(2, int(0))

minMask = 63193088
minTitle = ''

maxMask = 0
maxTitle = ''


for file in fileSet:
    image = cv.imread(os.path.join(path, file), 0)
    
    unique, counts = np.unique(image, return_counts=True)
    
    i = 0
    for un in unique:
        j = 0
        if un > 0:
            j = 1
        if un > 200:
            j = 2
        
        if un > 200 and minMask > counts[i] :
            minMask = counts[i]
            minTitle = file
        
        if un > 200 and maxMask < counts[i] :
            maxMask = counts[i]
            maxTitle = file
        
        
        if un > 10 and un < 250:
            print(file)
        
        types[j] += counts[i]
        total += counts[i]
        i += 1
    
print(total)
print(types)

types = types / total
print(types)


print(minMask)
print(minTitle)
print(minMask / (512*608))

print(maxMask)
print(maxTitle)
print(maxMask / (512*608))