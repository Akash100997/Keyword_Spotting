import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import librosa as lb
import numpy as np


TRAINED_MODEL = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 16000


def preprocess(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    # Load the audio file
    signal, sr = lb.load(file_path)

    # Ensure consistency in Audio File
    if len(signal) >= NUM_SAMPLES_TO_CONSIDER:
        signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # Extract the MFCCs
        mfccs = lb.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return mfccs.T


class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "down",
        "eight",
        "go",
        "happy",
        "house",
        "marvin",
        "nine",
        "right",
        "seven",
        "sheila"
    ]

    _instance = None

    def predict(self, file_path):
        # Extract the MFCCs
        MFCCs = preprocess(file_path)

        # Convert 2D MFCCs to 4D
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # Make Predictions
        prediction = self.model.predict(MFCCs)
        predicted_index = np.argmax(prediction)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


def Keyword_spotting_service():
    # Ensure that we only have one instance
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(TRAINED_MODEL)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_spotting_service()

    keyword1 = kss.predict("Test\\happy.wav")
    keyword2 = kss.predict("Test\\marvin.wav")
    keyword3 = kss.predict("Test\\eight.wav")

    print(f"Predicted Keyword: {keyword1}, {keyword2}, {keyword3}")
