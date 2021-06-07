# **Final-Project : Speech and text classification for sentiment analysis**
## Introduction
Text and speech classification models are the future in terms of mental health illness diagnosis as depression. According to the World Health Organization (WHO), 322 million people worldwide suffer from depression. The objective of this investigation is to analyze some of the newest models for classification to determine whether they are a good tool that doctors and therapists could use un a near future.  

## Methods
### Dataset

For the datasets we used IEMOCAP that contains the audio files needed for speech analysis as well as its transcription, thanks to this features we were able to use this dataset for both speech and text analysis. We also used EmoDB dataset for speech analysis.

### Classification
For speech analysis we used EmoAudioNet vectorizer method by extracting the MFCC vector from the audio this will later on pass on to a BiLSTM + Attention model whcich will generate a prediction.
The models used for text classification were BERT and LSTMs, we used this models as we consider we would get results with good accuracy. We used  Pytorch as our open source machine learning library for BERT as well as Keras for  EmoAudioNet and for our LSTM + BERT text analysis.


## Results
### BiLSTM + Attention
After 150 epochs in speech analysis we ended up with a 76% accuracy a great result for our speech model.

### BERT
For BERT we got a 89% accuracy, despite this fact some of the emotions had a lower accuray, this could be caused by our dataset label distribution.

### BERT+LSTM
After getting those values with Bert we decided to combine such model with an LSTM to obtain better results, at the end of our training we ended up with a 94% accuracy actually improving our previous model.

## Conclusion
Despite having some trouble at the beginig with EmoAudioNet we ended up getting great accuracy of 84%. For text classification with BERT and LSTMs the results were great as well 89% and 94% accuracy.
This models can be used in the future for an acurrate sentiment classification that could help specialists in mental health areas.


## How to run the code

### BiLSTM + Attention
In the ColabTrainingFiles folder we can find the BiLSTM + Att folder where you can find the code to train the model which could also be found on the PreTrainedModels folder. After you train the model with the Train.ipynb notebook two files will be generated, the best_model.h5 and the scaler.pickle files that will be necesary to test the model. In the ColabTestFiles you can find the necesary files to test the model which you need to run the main.py file which will ask you for inputs this inputs could be text, audio, or both to generate a prediction.
### BERT + LSTM
El dataset utilizado es IEMOCAP y se encuentra en la carpeta data--> text_audio.csv
### BERT 
El dataset para correr el clasificador de sentimientos se encuentra en el folder data-->DATASET_PROYECTO_NLP.csv el cual también pertenece al dataset de IEMOCAP.

In the next url we can find the checkpoint for the bert model
checkpoints: https://drive.google.com/drive/folders/1r0hBIOWnKkLzQnUXplr5D_HPLFBCmSeT?usp=sharing
