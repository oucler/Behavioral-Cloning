# Behavioral-Cloning P3
## Description
In this project, collecting data of driving in a simulator then feeding the data (camera view and steering) into a neural network to clone behaivor of the driver. Camera view includes center, left, and right images with only one steering data then steering is adjusted depending on the which left or right view. 

To avoid overfitting and removing unrelated information, images goes through preprocessing stages.

As a convulution network model, the model described in [NVDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) paper is used. 

1. Submission includes all required files and can be used to run the simulator in autonomous mode
Two new files are generated model.py and helper.py. In model.py file model architecture is defined and trainig is also done in model.py. Images need to be augmented to avoid overfitting. 

* model-000.h5 is created if the model is better than previous one
*

## Neural Network Model Specificatios 
The model table shown below is based on [NVDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and some modification made to the model -- images contain information and to avoid overfitting 50% of data is dropped. 

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
| Dense_1(Dense)       | None,100        |  844900 |
| Dense_2(Dense)       | None,50         |   5050  |
| Dense_3(Dense)       | None,10         |    510  |
| Dense_4(Dense)       | None,1          |    11   |
