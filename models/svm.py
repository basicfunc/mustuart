import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import sys


def load_song(filename):
    rate, signal = wavfile.read(filename)
    mfcc_features = mfcc(signal, rate, numcep=20)
    mfcc_features = mfcc_features.mean(axis=0)
    return mfcc_features.reshape(1, -1)


if len(sys.argv) != 2:
    print("Error: You forgot to pass song as argument.")
    exit()

song_path = sys.argv[1]
song_path = song_path.strip()

if os.path.exists('saved_models/svm_model.pkl') and \
        os.path.exists('saved_models/data.npy') and \
        os.path.exists('saved_models/labels.npy'):

    print("Found saved model, loading..")

    with open('saved_models/svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    data = np.load('saved_models/data.npy')
    labels = np.load('saved_models/labels.npy')

    print(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Load a song to predict its genre
    # mfcc_features = scaler.transform(load_song(song_path))

    # Predict the genre of the song
    # genre = clf.predict(mfcc_features)

    print("The genre of the song is", *
          clf.predict(scaler.transform(load_song(song_path))))

else:
    print("Unable to found saved model, training..")
    # Set the path to the directory containing the audio files
    audio_dir = 'dataset/genres'

    # Define the list of genres
    genres = ['blues', 'classical', 'country', 'disco',
              'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Load the audio data and extract MFCC features for each file
    data = []
    labels = []
    for genre in genres:
        genre_dir = os.path.join(audio_dir, genre)
        for filename in os.listdir(genre_dir):
            filepath = os.path.join(genre_dir, filename)
            rate, signal = wavfile.read(filepath)
            mfcc_features = mfcc(signal, rate, numcep=20)
            mfcc_features = mfcc_features.mean(axis=0)
            data.append(mfcc_features)
            labels.append(genre)

    # Convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    np.save('saved_models/data.npy', data)
    np.save('saved_models/labels.npy', labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the SVM classifier
    clf = svm.SVC(C=1, degree=2, gamma='scale', kernel='linear')

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    with open('saved_models/svm_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # Predict the genres of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Load a song to predict its genre
    mfcc_features = scaler.transform(load_song(song_path))

    # Predict the genre of the song
    genre = clf.predict(mfcc_features)
    print("The genre of the song is", genre[0])
