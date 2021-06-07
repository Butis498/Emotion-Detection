from ModelBILSTM import ModelBiLSTM


def main():

    type_of_input = input('Text(0) Audio(1) Text and Audio(2')
    audio_model = ModelBiLSTM()
    audio_model.load_model('best_model.h5','scaler.pickle')

    if type_of_input == 0:
        text = input('Type your input: ')
        # Add bert prediction 
    elif type_of_input == 1:
        audio_file = input('Type wav file name: ')
        prediction = audio_model.predict(audio_file)
        print(prediction)

    elif type_of_input == 2:
        text = input('Type your input: ')
        audio_file = input('Type wav file name: ')
        prediction_audio = audio_model.predict(audio_file)
        # Add bert prediction 

        #join predictions



if __name__=="__main__":
    main()