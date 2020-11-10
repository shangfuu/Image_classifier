# Train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import argparse
import time
import numpy as np
from config import pkl_feature, pkl_label, model_name, log_name

def save_model(fname, model):
    model.save(fname)

def simple_model(Data_feature, Data_lable):
    # scale
    Data_feature = Data_feature / 255.0
    # Modeling
    model = Sequential()

    # First Layer
    model.add(Conv2D(64, (3, 3), input_shape=Data_feature.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second Layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Third Layer
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    # Output Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    # TensorBoard (log)
    tensorboard = TensorBoard(log_dir="logs/{}".format(log_name))

    # Training
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.fit(Data_feature, Data_lable, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
    save_model(model_name, model)


def hard_model(Data_feature, Data_lable):
    ## Version 2 of Model for more accurancy ###
    ## But CPU Run too slow need to speed up by GPU ###
    layer_sizes = [32, 64, 128]
    conv_layers = [1, 2, 3]
    dense_layers = [0, 1, 2]

    # Do every layer
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                # Name of data
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                print(NAME)

                ## Modeling ##
                model = Sequential()

                # First Layer
                model.add(Conv2D(layer_size, (3, 3), input_shape=Data_feature.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # Second ( conv_layer ) Layer
                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                # Third Layer
                model.add(Flatten())
                # (Dense_Layer)
                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))
                
                # Output Layer
                model.add(Dense(1))
                model.add(Activation('sigmoid'))
                
                # Tensorboard (logs)
                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
                
                # Training
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(Data_feature, Data_lable, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

    save_model(model_name, model)

def parse_arg():
    parser = argparse.ArgumentParser(description='Train cat/dog image classification model')
    parser.add_argument('-m', '--model', nargs='?', default="simple")
    args = parser.parse_args()
    return args, parser

if __name__ == '__main__':
    # load features and labels
    # Data_feature = np.array(pickle.load(open(pkl_feature, "rb")))
    # Data_lable = np.array(pickle.load(open(pkl_label, "rb")))
    Data_feature = pickle.load(open(pkl_feature, "rb"))
    Data_lable = pickle.load(open(pkl_label, "rb"))

    # print(len(Data_feature))
    # print(len(Data_lable))

    args, parser = parse_arg()
    if args.model == 'simple':
        simple_model(Data_feature, Data_lable)
    elif args.model == 'hard':
        hard_model(Data_feature, Data_lable)
    else:
        parser.error('modle: hard/simple')

    


