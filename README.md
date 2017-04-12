# Behavioral-Cloning P3
## Description
In this project, collecting data of driving in a simulator then feeding the data (camera view and steering) into a neural network to clone behaivor of the driver. Camera view includes center, left, and right images with only one steering data then steering is adjusted depending on the which left or right view. 

To avoid overfitting and removing unrelated information, images goes through preprocessing stages.

As a convulution network model, the model described in [NVDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) paper is used. 



=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================

## NVDIA Model 

| Layer(type)          | Output Shape    | Param#  |
| ---------------------|:---------------:| -------:|
| Lambda               | None,75,320,3   |    0    |
| Cropping(Cropping2D) | None,90,320,3   |    0    |
| Conv2d_1(Conv2D)     | None,36,158,24  |   1824  |
| Conv2d_2(Conv2D)     | None,16,77,36   |  21636  |
| Conv2d_3(Conv2D)     | None,6,37,48    |  43248  |
| Conv2d_4(Conv2D)     | None,4,35,64    |  27712  |
| Conv2d_5(Conv2D)     | None,2,33,64    |  36928  |
| Dropout_1(Dropout)   | None,2,33,64    |    0    |
| Flatten_1(Flatten)   | None,4224       |    0    |
