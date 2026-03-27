import librosa
import numpy as np

def extract_features(file):
    audio, sr = librosa.load(file, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1)
