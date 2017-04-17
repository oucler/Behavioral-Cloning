"""
/// \ Author: Onur Ucler
/// \ Definition: Two classes and main function are implemented
/// \             build_model(generator) class has the model specifications also inherits from generator class
/// \             train_model(build_model) class trains and validates data defined in build_model (inherited)
/// \             main() provides the list of the command options and initiates the flow 
/// \ How to run: Instrantiate train_model class then call train() function and train_model class inherits from build_model
/// \             tm = train_model(args)
/// \             tm.train()
"""
import argparse
from helper import generator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Cropping2D, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import matplotlib.pyplot as plt 
#/// \ A Deep Learing Model specifications are defined in build_model class
#/// \ build_model() class inherits from generator class in helper module  
class build_model(generator):
    def __init__(self,args):
        generator.__init__(self,args)
        self.keep_prob = args.keep_prob
    def model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0,input_shape = self.augment_shape))
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Dropout(self.keep_prob))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1)) 
        model.summary()
        return model
#/// \ train_model class recieves data from batch_generator in generator class
#/// \ and fit data into a model if the accuracy better than previous model then
#/// \ new model will be saved. 
#/// \ Loss function plot shows loss trend between training and validation set
class train_model(build_model):
    def __init__(self,args):
        build_model.__init__(self,args)
        self.learning_rate = args.learning_rate
        self.model_name = args.model_name
        self.epoch = args.epoch
        self.steps_per_epoch = args.steps_per_epoch
    def train(self):
        X_train_loc,X_valid_loc,y_train,y_valid = self.load_image()
        #self.read_image(X_train_loc[0])
        
        m_checkpoint = ModelCheckpoint(self.model_name,
                        monitor='val_loss',
                        verbose=0,
                        save_best_only=True,
                        mode='auto')
        model = self.model()
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        history_obj = model.fit_generator(self.batch_generator(X_train_loc,y_train),
                        self.steps_per_epoch,
                        self.epoch,
                        max_q_size=1,
                        validation_data=self.batch_generator(X_valid_loc,y_valid),
                        nb_val_samples=len(X_valid_loc),
                        callbacks=[m_checkpoint],
                        verbose=1)

        ### plot the training and validation loss for each epoch
        plt.plot(history_obj.history['loss'])
        plt.plot(history_obj.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
#/// \  Default arguments and command line options for model.py is provided        
def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning Project 3")
    parser.add_argument('-d', help='image directory', dest='image_dir',type=str,default='data\data\IMG')
    parser.add_argument('-bs', help='batch size', dest='batch_size', type=int, default=40)    
    parser.add_argument('-k',help='dropping percantage of data',dest='keep_prob', type=int,default=0.5)    
    parser.add_argument('-lr',help='learning rate',dest='learning_rate',type=int,default=1.0e-4) 
    parser.add_argument('-mn',help='model name',dest='model_name',type=str,default='model-6labs_trial2.h5')
    parser.add_argument('-e',help='number of epoch',dest='epoch',type=int,default=10)
    parser.add_argument('-spe',help='steps per epoch',dest='steps_per_epoch',type=int,default=2000)
    args = parser.parse_args()
    
    tm  = train_model(args)
    tm.train()

if __name__ == '__main__':
    main()


