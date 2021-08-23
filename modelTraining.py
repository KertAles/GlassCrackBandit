# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 21:02:48 2021

@author: Kert PC
"""

import os

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import ndimage
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import pandas as pd
from focal_loss import SparseCategoricalFocalLoss


import PIL
from PIL import ImageOps
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import random
import math
from enum import Enum


class InputType(Enum) :
    AVERAGE = "average"
    FOUR_CHANNEL = "four_channel"
    STOKES = "stokes"
    STOKES_CALC = "stokes_calc"
    STOKES_CALC_PLUS = "stokes_calc_plus"


class WindowImages(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
    
    def deinterlace(self, path) :
        image = np.array(load_img(path, color_mode="grayscale"))
        
        
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
                x = np.array(list(range(0, w, d))) + j
                y = np.array(list(range(0, h, d))) + j
        
                subImgs[j] = np.expand_dims(image[np.ix_(y,x)], axis=-1)
        else :
            print('Picture of invalid size')
            
        return subImgs
    
    def extract_stokes(self, channels) :
        h = channels[0].shape[0]
        w = channels[0].shape[1]
        
        stokes = np.zeros((h, w, 3))
        stokes[:, :, 0] = (channels[0][:, :, 0] +  channels[1][:, :, 0]) / 2
        stokes[:, :, 1] = (channels[0][:, :, 0] -  channels[1][:, :, 0] + 255) / 2
        stokes[:, :, 2] = (channels[2][:, :, 0] -  channels[3][:, :, 0] + 255) / 2
        
        return stokes
    
    def calculateDegAngOfPol(self, stokes, w_s = 0.75) :
        h = stokes.shape[0]
        w = stokes.shape[1]
        
        mean_S0 = np.mean(stokes[:, :, 0])
        S0_adj = stokes[:, :, 0] * w_s + mean_S0 * (1-w_s)
        dolp = (np.sqrt(np.square(stokes[:, :, 1]) + np.square(stokes[:, :, 2])) / S0_adj)
        
        mean_S1 = np.mean(stokes[:, :, 1])
        S1_adj = stokes[:, :, 1] * w_s + mean_S1 * (1-w_s)
        aolp = 0.5 * np.arctan2(stokes[:, :, 2], S1_adj)
        
        
        d_a_olp = np.zeros((h, w, 2))
        d_a_olp[:, :, 0] = dolp
        d_a_olp[:, :, 1] = aolp
        
        return d_a_olp
    
    def image_histogram_equalization(self, channel, number_bins=256):
        image_histogram, bins = np.histogram(channel.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize
    
        # use linear interpolation of cdf to find new pixel values
        channel_equalized = np.interp(channel.flatten(), bins[:-1], cdf)
    
        return channel_equalized.reshape(channel.shape), cdf
    
    

    def __init__(self, images, masks, input_type=InputType.AVERAGE, batch_size=4, img_size=(512,608), augment=True):
        self.images = images
        self.masks = masks
        
        self.input_type = input_type
        
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.augment = augment
        
    def __len__(self):
        if self.augment :
            ret = len(self.images * 3) // self.batch_size  
        else :
            ret = len(self.images) // self.batch_size
        
        return ret

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        
        aug_round = i // len(self.images)
        
        if aug_round > 2:
            aug_round = 2
        
        if i >= len(self.images) :
            i = i % len(self.images)
            
        
        batch_images = self.images[i : i + self.batch_size]
        batch_masks = self.masks[i : i + self.batch_size]
        
        return self.getData(batch_images, batch_masks, aug_round=aug_round)
        
    
    def applyAugmentation(self, img) :
        aug_option = math.floor(random.Random().random() * 3)
        
        if aug_option == 0 :
            noise = np.random.normal(0, .10, img.shape) * 255
            img = img + noise
          
        elif aug_option == 1 :
            img = ndimage.gaussian_filter(img, sigma=2)
        
        elif aug_option == 2 :
            img_eq = np.zeros(img.shape)
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                img_eq[:, :, i] = self.image_histogram_equalization(channel)[0]
            img = img_eq
            
        return img

    
    def getAllData(self) :
        return self.getData(self.images, self.masks)

    
    def getData(self, batch_images, batch_masks, aug_round=0) :
        if self.input_type == InputType.AVERAGE :
            num_of_channels = (1,)
        elif self.input_type == InputType.FOUR_CHANNEL :
            num_of_channels = (4,)
        elif self.input_type == InputType.STOKES :
            num_of_channels = (3,)
        elif self.input_type == InputType.STOKES_CALC :
            num_of_channels = (2,)
        elif self.input_type == InputType.STOKES_CALC_PLUS :
            num_of_channels = (3,)
     
        x = np.zeros((len(batch_images),) + self.img_size + num_of_channels, dtype="float32")
  
        for j, path in enumerate(batch_images):
            subImgs = self.deinterlace(path)
            
            if self.input_type == InputType.AVERAGE :
                img = (subImgs[0] + subImgs[1] + subImgs[2] + subImgs[3]) / 4
            elif self.input_type == InputType.FOUR_CHANNEL :
                img = np.concatenate((subImgs[0], subImgs[1], subImgs[2], subImgs[3]), axis=-1)
                
            if self.input_type == InputType.AVERAGE or self.input_type == InputType.FOUR_CHANNEL :
                if aug_round == 1 :
                    img = np.flipud(img)
                    img = self.applyAugmentation(img)
                    
                elif aug_round == 2 :
                    img = np.fliplr(img)
                    img = self.applyAugmentation(img)
            else :
                if aug_round == 1 :
                    subImgs = np.flipud(subImgs)
                    subImgs = self.applyAugmentation(subImgs)
                    
                elif aug_round == 2 :
                    subImgs = np.fliplr(subImgs)
                    subImgs = self.applyAugmentation(subImgs)
                
            if self.input_type == InputType.STOKES :
                subImgs = ndimage.gaussian_filter(subImgs, sigma=0.7)
                img = self.extract_stokes(subImgs)
            elif self.input_type == InputType.STOKES_CALC :
                subImgs = ndimage.gaussian_filter(subImgs, sigma=0.7)
                stokes = self.extract_stokes(subImgs)
                
                img = self.calculateDegAngOfPol(stokes)
            elif self.input_type == InputType.STOKES_CALC_PLUS :
                gray = (subImgs[0] + subImgs[1] + subImgs[2] + subImgs[3]) / 4
                
                subImgs = ndimage.gaussian_filter(subImgs, sigma=0.7)
                stokes = self.extract_stokes(subImgs)
                
                img = np.concatenate((gray, self.calculateDegAngOfPol(stokes)), axis=-1)
                
                
            if self.input_type == InputType.FOUR_CHANNEL:
                mean = np.mean(img)
                std = 3 * np.std(img)
                    
                img = (img - (mean - std))
                img = ((img / (mean + std)) * 2) - 1 
            else :
                for i in range(img.shape[2]) :
                    mean = np.mean(img[:, :, i])
                    std = 2 * np.std(img[:, :, i])
                    
                    img[:,:,i] = (img[:,:,i] - (mean - std))
                    img[:,:,i] = ((img[:,:,i] / (mean + std)) * 2) - 1
                    #print('POST ::: Mean: ' + str(np.mean(img[:, :, i])) + ' | Min: ' + str(img[:, :, i].min()) + ' | Max: ' + str(img[:, :, i].max())  + ' | Std: ' + str(np.std(img[:, :, i])) )
            
            #print('ALLTOGETHER ::: Mean: ' + str(np.mean(img)) + ' | Min: ' + str(img.min()) + ' | Max: ' + str(img.max())  + ' | Std: ' + str(np.std(img)) )
            
            x[j] = img
            
            
        y = np.zeros((len(batch_images),) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_masks):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            
            if aug_round == 1 :
                img = np.flipud(img)
            elif aug_round == 2 :
                img = np.fliplr(img)
            
                
            img = np.expand_dims(img, axis=-1) / 255
            
            #print('Mean: ' + str(np.mean(img)) + ' | Min: ' + str(img.min()) + ' | Max: ' + str(img.max()))
            
            y[j] = img
        return x, y

def unet_model_blocks(inputs=None, num_classes=2, input_type=InputType.AVERAGE, block_number=4, filter_number=16):
        if inputs is None:
            if input_type == InputType.AVERAGE :
                num_of_channels = (1,)
            elif input_type == InputType.FOUR_CHANNEL :
                num_of_channels = (4,)
            elif input_type == InputType.STOKES :
                num_of_channels = (3,)
            elif input_type == InputType.STOKES_CALC :
                num_of_channels = (2,)
            elif input_type == InputType.STOKES_CALC_PLUS :
                num_of_channels = (3,)
            
            inputs = layers.Input((None, None) + num_of_channels)
            
        filter_num = filter_number
        x = inputs
        block_features = []
        for i in range(block_number):
            fn_cur = filter_num*(2**(i))
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
            conv1 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv1)
            conv1 = Dropout(0.2)(conv1)
            conv1 = BatchNormalization()(conv1)
            block_features.append(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fn_cur = filter_num*(2**(block_number))
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(x)
        conv3 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.2)(conv3)
        x = drop3
        for i in range(block_number):
            fn_cur = filter_num*(2**(block_number - i - 1))
            up8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([block_features.pop(), up8], axis=3)

            conv8 = Conv2D(fn_cur, (3, 3), activation="relu", padding="same")(merge8)
            conv8 = Conv2D(fn_cur, (3, 3), activation='relu', padding='same')(conv8)
            conv8 = Dropout(0.2)(conv8)
            conv8 = BatchNormalization()(conv8)

            x = conv8

        conv10 = Conv2D(num_classes, (1,1), activation='softmax', padding="same")(x)
        
        model = keras.Model(inputs, conv10)

        return inputs, conv10, model
    

cluster_mode = True

if cluster_mode :
    input_dir = '/storage/local/hdd/dataset/'
    target_dir = '/storage/local/hdd/masks_renamed/'
    model_dir = '/home/ales/gcb/GlassCrackBandit/models/'
else :   
    input_dir = 'F:/Diploma/dataset/'
    target_dir = 'F:/Diploma/masks_renamed/'
    model_dir = 'F:/Diploma/models/'

if cluster_mode :
    os.environ["CUDA_VISIBLE_DEVICES"]="1"


build_model = True
calculate_metrics = True
show_predictions = True
model_path = 'F:/Diploma/code/models/model_stokes_calc_1'

augment = True

img_size = (512, 608)
#img_size = (128, 152)
num_classes = 2
batch_size = 12
num_epochs = 35

input_type = InputType.FOUR_CHANNEL

images = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".bmp")
    ]
)
masks = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".bmp")
    ]
)

print("Number of samples:", len(images))

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Split our img paths into a training and a validation set
val_samples = 43
random.Random(1337).shuffle(images)
random.Random(1337).shuffle(masks)

train_images = images[:-val_samples]
train_masks = masks[:-val_samples]
val_images = images[-val_samples:]
val_masks = masks[-val_samples:]


# Instantiate data Sequences for each split
train_gen = WindowImages(train_images, train_masks, input_type=input_type, batch_size=batch_size, img_size=img_size, augment=augment)
val_gen = WindowImages(val_images, val_masks, input_type=input_type, batch_size=batch_size, img_size=img_size, augment=augment)

"""
imgs = train_gen.__getitem__(0)

fig = plt.figure(figsize=(60., 60.))
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes
                 )

for ax, j in zip(grid, range(9)) :
    img = imgs[0][j//3]
    ax.imshow(img[:,:,j%3])
      
"""



"""
#data_gen = WindowImages(images, masks, input_type=input_type, batch_size=batch_size, img_size=img_size)
tf.config.experimental.set_per_process_memory_fraction(0.75)
tf.config.experimental.set_per_process_memory_growth(True)
"""
if True:
    if build_model :
        # Build model
        #model = get_model(img_size, num_classes, input_type=input_type)
        
        inputs, outputs, model = unet_model_blocks(input_type=input_type, block_number=4, filter_number=16)
            
        #model.summary()
        model.compile(optimizer="adam", loss=SparseCategoricalFocalLoss(gamma=3), metrics=["sparse_categorical_accuracy"])
            
        # Train the model, doing validation at the end of each epoch.
        epochs = num_epochs
        
        callbacks = [
            keras.callbacks.ModelCheckpoint("window_segmentation", save_best_only=True)
        ]
            
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

        
        model_names = [
            mod_name
            for mod_name in os.listdir(model_dir)
            if input_type.value in mod_name
        ]
        
        model_type_num = str(len(model_names) + 1)
        model_path = model_dir + 'model_' + input_type.value + '_' + model_type_num
        
        model.save(model_path)
        
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        
        # save to json:  
        hist_json_file = model_dir + 'history_' + input_type.value + '_' + model_type_num + '.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
                
        
    else :
        model = keras.models.load_model(model_path)
        
        #val_gen = WindowImages(val_images, val_masks, input_type=input_type, batch_size=batch_size, img_size=img_size, augment=False)
    
       
        
       # results = model.evaluate(val_gen)
       # print("test loss, test acc:", results)
    
    
    # Generate predictions for all images in the validation set
    
    val_gen = WindowImages(val_images, val_masks, input_type=input_type, batch_size=1, img_size=img_size, augment=False)
    #val_gen = WindowImages(train_images, train_masks, input_type=input_type, batch_size=1, img_size=img_size, augment=False)
    
            
    if show_predictions and not cluster_mode :
        def get_mask(i):
            mask = np.argmax(val_preds[i], axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
            #display(img)
            return img
        
        def get_metrics(i):
            mask = np.argmax(val_preds[i], axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            _, gt = val_gen.__getitem__(i)
            
            gt = gt[0]
        
            TP = 0
            FP = 0
            TN = 0
            FN = 0
        
            for i in range(gt.shape[0]): 
                for j in range(gt.shape[1]): 
                    if gt[i][j] == mask[i][j] == 1:
                       TP += 1
                    if mask[i][j] == 1 and gt[i][j] != mask[i][j]:
                       FP += 1
                    if gt[i][j] == mask[i][j] == 0:
                       TN += 1
                    if mask[i][j] == 0 and gt[i][j] != mask[i][j]:
                       FN += 1
                       
            pr = TP / (TP + FP)
            re = TP / (TP + FN)
            f1 = 2 / (1/pr + 1/re)
        
            return (pr, re, f1)
        
        # Display results for validation image #10
        i = 0
        num_of_vals = 25
        
        val_preds = model.predict(val_gen)
        
        
        fig = plt.figure(figsize=(80., 80.))
        grid = ImageGrid(fig, 111, 
                         nrows_ncols=(num_of_vals, 3),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes
                         )
        
        for ax, j in zip(grid, range(i, i + num_of_vals * 3)) :
            if j%3 == 0 :
                img = load_img(val_images[j // 3], target_size=img_size)
                ax.imshow(img)
            elif j%3 == 1 :
                img = PIL.ImageOps.autocontrast(load_img(val_masks[j // 3]))
                ax.imshow(img)
            else :
                ax.imshow(get_mask(j // 3))
                

        if calculate_metrics :
            
            num_of_preds = len(val_preds)
            pr_sum = 0.0
            re_sum = 0.0
            f1_sum = 0.0
            
        
            for i in range(num_of_preds):
   
                (pr, re, f1) = get_metrics(i)
            
                pr_sum += pr
                re_sum += re
                f1_sum += f1
                
                print('Precision: ' + str(pr) + ' ;  ' + 'Recall: ' + str(re) + ' ;  ' + 'F1: ' + str(f1))
        
            
            print('AVERAGE ::: Precision: ' + str(pr_sum / num_of_preds) + ' ;  ' + 'Recall: ' + str(re_sum / num_of_preds) + ' ;  ' + 'F1: ' + str(f1_sum / num_of_preds))
    
