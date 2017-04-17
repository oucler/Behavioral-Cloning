"""
/// \ Author: Onur Ucler
/// \ Definition: Two classes, image_processing and generator.
/// \             generator class inherits from image_processing class.
/// \             Images are modified by image_processing class 
/// \ How to run: Helper function is used for generating bacthes with modified images
/// \             from helper import generator -- functionality of the helper functions imported 
"""
import os
import matplotlib.image as m_image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
np.random.seed(0)

#/// \  Base class is used for processing and modifying images
class image_processing:
    def __init__(self,args):
        self.image_dir = args.image_dir
        #self.augment_shape = (75,320,3)
        self.augment_shape = (66,200,3)
        self.original_shape = (160,320,3)
        self.range_x = 100
        self.range_y = 10
 
    def load_image(self):
        df_pd = pd.read_csv(os.path.join(self.image_dir,'..\driving_log.csv'))
        X_loc = df_pd[['center','left','right']].values 
        y_loc = df_pd['steering'].values
        X_train_loc,X_valid_loc,y_train_loc,y_valid_loc = train_test_split(X_loc,y_loc,test_size=0.2,random_state=42)  
        return X_train_loc,X_valid_loc,y_train_loc,y_valid_loc 
    def read_image(self,img_loc):
        center_img,left_img,right_img = img_loc
        center_img = m_image.imread(os.path.join(self.image_dir, center_img.split('/')[-1]))
        left_img = m_image.imread(os.path.join(self.image_dir, left_img.split('/')[-1]))
        right_img = m_image.imread(os.path.join(self.image_dir, right_img.split('/')[-1]))
        #self.shape = center_img.shape
        return center_img,left_img,left_img
    def pick_camera_view(self,X_loc,y):
        center_img, left_img, right_img = self.read_image(X_loc)
        selection = np.random.choice(3)
        random = np.random.rand()
        if selection == 0 and random >= 0.5:
            return left_img, y + 0.25
        elif selection == 1 and random >= 0.5:
            return right_img, y - 0.25
        else:
            return center_img, y  
    def random_brightness(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 
    def random_translate(self,image,steering):
        trans_x = self.range_x * (np.random.rand() - 0.5)
        trans_y = self.range_y * (np.random.rand() - 0.5)
        steering += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image,steering    
    
    def random_shadow(self,image):
        x1, y1 =  self.original_shape[1]* np.random.rand(), 0
        x2, y2 = self.original_shape[1] * np.random.rand(), self.original_shape[0]
        xm, ym = np.mgrid[0:self.original_shape[0], 0:self.original_shape[1]]
        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)        
        
    def rgb2yuv(self,image):
        return cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    def flip_image(self,image,steering):
        if np.random.rand() <= 0.5:
            #image = np.fliplr(image)            
            image = cv2.flip(image,1)
            steering = -steering
        return image,steering
    def resize_image(self,image):
        return cv2.resize(image,(self.augment_shape[1],self.augment_shape [0]),cv2.INTER_AREA)
    def crop_image(self,image):
        return np.array(image[60:-25,:,:])
    def preprocessing(self,image):
        image = self.crop_image(image)
        image = self.resize_image(image)
        image = self.rgb2yuv(image)
        return image
    def augment_image(self,X_loc,y):
        image, steering = self.pick_camera_view(X_loc,y)
        image,steering = self.flip_image(image,steering)
        image,steering = self.random_translate(image,steering)
        image = self.random_shadow(image)
        image = self.random_brightness(image)

        return image, steering  
#/// \ Images and steering associated with send by batch size    
class generator(image_processing):
    def __init__(self,args):
        image_processing.__init__(self,args)
        self.batch_size = args.batch_size

    def batch_generator(self,X_loc,y):
        X = np.empty([self.batch_size,self.augment_shape[0],self.augment_shape[1],self.augment_shape[2]])
        steerings = np.empty(self.batch_size)
        while True:
            count=0
            for i in np.random.permutation(X_loc.shape[0]):
                if np.random.rand() <= 0.5:
                    image,steering = self.augment_image(X_loc[i],y[i])
                else:
                    center_img,left_img,right_img = self.read_image(X_loc[i])
                    image = center_img
                    steering = y[i]
                image = self.preprocessing(image)
                X[count] = image
                steerings[count] = steering
                count += 1
                if count == self.batch_size:
                    break
            yield X,steerings

    
