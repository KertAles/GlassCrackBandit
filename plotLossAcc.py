# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 00:35:38 2021

@author: Kert PC
"""

import json

import PIL
from PIL import ImageOps
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


file_path = 'F:/Diploma/code/models/history_four_channel_1.json'
file = open(file_path)

history = json.load(file)

if True:
        # summarize history for accuracy
        plt.plot(list(history['sparse_categorical_accuracy'].values()))
        plt.plot(list(history['val_sparse_categorical_accuracy'].values()))
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(list(history['loss'].values()))
        plt.plot(list(history['val_loss'].values()))
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()