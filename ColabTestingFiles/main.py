from ModelBILSTM import ModelBiLSTM
from BERT import EmotionsCommentTagger, predict_text

def join_predictions(text_pred,score_text, audio_pred,score_audio):
    labels = ['Anger', 'Happy', 'Sad', 'Neutral']

    
    if text_pred not in labels:
        return audio_pred
    
    else:
        if score_text> score_audio:
            return text_pred
        elif score_text < score_audio:
            return audio_pred
        
        else: return score_audio



def main():

    type_of_input = input('Text(0) Audio(1) Text and Audio(2')
    audio_model = ModelBiLSTM()
    audio_model.load_model('best_model.h5','scaler.pickle')
    text_classifier = predict_text()

    if type_of_input == 0:
        text = input('Type your input: ')
        text_classifier.predict() 
    elif type_of_input == 1:
        audio_file = input('Type wav file name: ')
        prediction = audio_model.predict(audio_file)
        print(prediction)

    elif type_of_input == 2:
        text = input('Type your input: ')
        audio_file = input('Type wav file name: ')
        prediction_audio,score_audio = audio_model.predict(audio_file)
        prediction_text,score_text = text_classifier.predict(text)
        pred = join_predictions(prediction_text,score_text, prediction_audio,score_audio)
        print(pred)

        



if __name__=="__main__":
    main()