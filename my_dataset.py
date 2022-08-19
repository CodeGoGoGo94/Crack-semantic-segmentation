# -*- coding: utf-8 -*-
"
import cv2
import os
import torch
from torch.utils.data import Dataset
from PIL import Image # 没有用openCV
import numpy as np
from torchvision import transforms
import albumentations as A
import unet
import random
from transformers import RandomHorizontalFlip
from matplotlib import pyplot as plt



class MyDataSet_train(Dataset): # 
    # 
    def __init__(self, path,transform):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'train_img')) # 
        self.transform = transform
      

    
    def __getitem__(self, index):
        img_name = self.name[index] 
        img_path = os.path.join(self.path,'train_img',img_name)
        label_path = os.path.join(self.path, 'train_lab',img_name.replace('jpg', 'png') )
        
        img = cv2.imread(img_path)
        
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE) #
        
        if self.transform is not None:
            transformed =  self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        
        img = img.astype('float32') / 255.0
        label = label.astype('float32') / 255.0
        
        
        trans = transforms.ToTensor()
        img = trans(img)
        label = trans(label)
        
        return img, label
        
    
    # 
    def __len__(self):
        return len(self.name)



class MyDataSet_test(Dataset): # 
    # 
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'test_img')) # 
        
    # 
    def __getitem__(self, index):
        img_name = self.name[index] 
        img_path = os.path.join(self.path,'test_img',img_name)
        label_path = os.path.join(self.path, 'test_lab',img_name.replace('jpg','png') )
        
        img = cv2.imread(img_path)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        
        img = img.astype('float32') / 255.0
        label = label.astype('float32') / 255.0
        
        trans = transforms.ToTensor()
        img = trans(img)
        label = trans(label)
        
        return img, label
        
    # 
    def __len__(self):
        return len(self.name)


class MyDataSet_500_train(Dataset): # 
    #
    def __init__(self, path,transform):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'train_img')) # 
        self.transform = transform
      
    # 
    
    def __getitem__(self, index):
        img_name = self.name[index] 
        img_name = img_name[0:-4]
        
        img_path = os.path.join(self.path,'train_img' , (img_name +'.jpg'))
        label_path = os.path.join(self.path, 'train_lab', (img_name + '.png'))
        
        img = cv2.imread(img_path)
        
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE) 
        
        if self.transform is not None:
            transformed =  self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        
        img = img.astype('float32') / 255.0
        label = label.astype('float32') / 255.0
        
        
        trans = transforms.ToTensor()
        img = trans(img)
        label = trans(label)
        
        return img, label
        
    
    def __len__(self):
        return len(self.name)


class MyDataSet_500_test(Dataset): 
    # 
   def __init__(self, path):
       self.path = path
       self.name = os.listdir(os.path.join(path, 'test_img')) # 
   
   
   def __getitem__(self, index):
       img_name = self.name[index] 
       img_name = img_name[0:-4]
       
       img_path = os.path.join(self.path,'test_img' , (img_name +'.jpg'))
       label_path = os.path.join(self.path, 'test_lab', (img_name + '_mask' + '.png'))
       
       img = cv2.imread(img_path)
       
       label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE) 
       
       img = img.astype('float32') / 255.0
       label = label.astype('float32') / 255.0
       
       
       trans = transforms.ToTensor()
       img = trans(img)
       label = trans(label)
       
       return img, label

  
   def __len__(self):
       return len(self.name)


class MyDataSet_CFD_train(Dataset):
   
    def __init__(self, path,transform):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'train_img')) # 
        self.transform = transform
      
    
    def __getitem__(self, index):
        img_name = self.name[index] 
        img_name = img_name[0:-4]
        
        img_path = os.path.join(self.path,'train_img' , (img_name +'.jpg'))
        label_path = os.path.join(self.path, 'train_lab', (img_name + '.png'))
        
        img = cv2.imread(img_path)
        
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE) 
        
        if self.transform is not None:
            transformed =  self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        
        img = img.astype('float32') / 255.0
        label = label.astype('float32') / 255.0
        
        
        trans = transforms.ToTensor()
        img = trans(img)
        label = trans(label)
        
        return img, label
        
    
    
    def __len__(self):
        return len(self.name)


