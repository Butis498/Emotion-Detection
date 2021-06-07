import numpy as np 
import pandas as pd
import tensorflow as tf
from vad_mfcc import extract_coef
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from AttentionClass import Attention
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle



class ModelBiLSTM():


    def __init__(self):
        self.model = None
        self.scaler = None


    def get_vector(self,wav_file,directory):
        input_data = np.ones([1, 60])
        try:                

            matrix = extract_coef(directory, wav_file)
            io = [matrix[:,:60]]
            io = np.array(io, dtype=np.float64)
            input_data = np.append(input_data, io, axis=0)
            

        except ValueError:
            print("Error obtaining the coef")

        return input_data


    def load_model(self,weights,scaler):
        self.model = self.create_model()
        self.model.load_weights(weights)
        with open(scaler, 'rb') as handle:
            self.scaler = pickle.load(handle)

    def create_model(self):
        model = Sequential()
        model.add(layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, dropout=0.25, recurrent_dropout=0.2),input_shape=(60, 1)))
        model.add(layers.Bidirectional(layers.LSTM(units=60, return_sequences=True, dropout=0.25, recurrent_dropout=0.2)))
        model.add(Attention(60))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        return model

    def predict(self,audio_file):

        path,fileName = os.path.split(audio_file)

        vector = self.get_vector(fileName,path)
        vector = self.scaler.transform(vector)
        vector = np.expand_dims(vector, -1)
        prediction = self.model.predict(vector)
        res = np.argmax(prediction, axis=1)
        labels = ['Happy', 'Neutral', 'Sad', 'Anger']
        res = labels[res]
        return res



        