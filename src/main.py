import wave
import pickle
import numpy as np
import sfeatpy as sp
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures


def dload_song(filename):
    fs, x = audioBasicIO.read_audio_file(filename)
    win_size = 0.1
    win_step = 0.05
    features, _, _ = ShortTermFeatures.feature_extraction(
        x, fs, win_size * fs, win_step * fs)
    mfccs = features[8:21, :]
    mfcc_features = mfcc.mean(axis=0)
    return mfcc_features.reshape(1, -1)


def load_song(song_path, sr=44100, win_size=0.1, win_step=0.1):
    # Load audio file
    [fs, x] = audioBasicIO.read_audio_file(song_path)

    # Resample if necessary
    if fs != sr:
        x = audioBasicIO.resample_signal(x, fs, sr)
        fs = sr

    # Check the shape of the audio data
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    elif x.ndim > 2:
        x = np.mean(x, axis=1, keepdims=True)

    # Extract short-term features
    features, _, _ = ShortTermFeatures.feature_extraction(
        x, sr, win_size * sr, win_step * sr)

    return features


def predict(song_path):
    print("Loading model...")

    with open('saved_models/svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    data = np.load('saved_models/data.npy')
    labels = np.load('saved_models/labels.npy')

    X_train, X_test, *_ = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Load a song to predict its genre
    mfcc_features = scaler.transform(load_song(song_path))

    # Predict the genre of the song
    genre = clf.predict(mfcc_features)

    return "".join(genre)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Bruh!!!")
        exit()

    song = sys.argv[1]
    print(predict(song))
