# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:08:45 2021

@author: Kert PC
"""

import os

path = 'F:/Diploma/masks_renamed/'
fileSet = os.listdir(path)

i = 1


for file in fileSet:
    baseFile = file.split('_')[0]
    newName = baseFile + '_gt_damage.bmp'
    
    os.rename(path + file, path + newName)
    
    i = i + 1
