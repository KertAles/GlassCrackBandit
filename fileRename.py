# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:38:34 2021

@author: Kert PC
"""

import os

path = 'F:/Diploma/dataset/'
fileSet = os.listdir(path)

i = 1

for file in fileSet: 
    newName = ('0000' + str(i))[-4:] + '.bmp'
    
    os.rename(path + file, path + newName)
    
    i = i + 1
    