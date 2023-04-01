import wave
import pickle
import numpy as np
import sfeatpy as sp
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def load_song(filename):
    with wave.open(filename, 'rb') as f:

        nchannels, _, _, nframes, *_ = f.getparams()

        frames = f.readframes(nframes)

        signal = np.frombuffer(frames, dtype=np.int16).astype(
            np.float32) / 32768.0

        if nchannels > 1:
            signal = signal[::nchannels]

    mfcc = sp.mfcc(signal)
    mfcc_features = mfcc.mean(axis=0)
    return mfcc_features.reshape(1, -1)


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
    print(predict(sys.argv[1]))
