
import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

if os.path.exists('models/svm_model.pkl'):
    with open('models/svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Load a song to predict its genre
    filename = 'Counting-Stars.wav'
    rate, signal = wavfile.read(filename)
    mfcc_features = mfcc(signal, rate, numcep=20)
    mfcc_features = mfcc_features.mean(axis=0)
    scaler = StandardScaler()
    mfcc_features = scaler.fit_transform(mfcc_features.reshape(1, -1))

    # Predict the genre of the song
    genre = clf.predict(mfcc_features)
    print("The genre of the song is", genre[0])


else:
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

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the SVM classifier
    clf = svm.SVC(kernel='linear')

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # Predict the genres of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Load a song to predict its genre
    filename = 'Counting-Stars.wav'
    rate, signal = wavfile.read(filename)
    mfcc_features = mfcc(signal, rate, numcep=20)
    mfcc_features = mfcc_features.mean(axis=0)
    mfcc_features = mfcc_features.reshape(1, -1)
    mfcc_features = scaler.transform(mfcc_features)

    # Predict the genre of the song
    genre = clf.predict(mfcc_features)
    print("The genre of the song is", genre[0])
