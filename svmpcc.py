import sys
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import numpy as np
import os


def load_song(filename, pca):
    rate, signal = wavfile.read(filename)
    signal = signal.astype('float32') / 32767  # normalize to [-1, 1]
    # zero-pad to multiple of 512
    signal = np.pad(signal, (0, 512 - signal.shape[0] % 512))
    frames = signal.reshape(-1, 512)
    features = []
    for frame in frames:
        # apply PCA to each frame
        feature = pca.transform(frame.reshape(1, -1))
        feature = feature.mean(axis=0)
        features.append(feature)
    return np.mean(features, axis=0).reshape(1, -1)


if len(sys.argv) != 2:
    print("Error: You forgot to pass song as argument.")
    exit()

song_path = sys.argv[1]
song_path = song_path.strip()

if os.path.exists('models/svm_model.pkl') and \
        os.path.exists('models/data.npy') and \
        os.path.exists('models/labels.npy'):

    with open('models/svm_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    data = np.load('models/data.npy')
    labels = np.load('models/labels.npy')

    # Apply PCA to the data
    pca = PCA(n_components=20)
    data = pca.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Load a song to predict its genre
    mfcc_features = scaler.transform(load_song(song_path, pca))

    # Predict the genre of the song
    genre = clf.predict(mfcc_features)
    print("The genre of the song is", *genre)


else:
    # Set the path to the directory containing the audio files
    audio_dir = 'datasets/genres'

    # Define the list of genres
    genres = ['blues', 'classical', 'country', 'disco',
              'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Load the audio data and extract PCA features for each file
    data = []
    labels = []
    for genre in genres:
        genre_dir = os.path.join(audio_dir, genre)
        for filename in os.listdir(genre_dir):
            filepath = os.path.join(genre_dir, filename)
            rate, signal = wavfile.read(filepath)
            signal = signal.astype('float32') / 32767  # normalize to [-1, 1]
            # zero-pad to multiple of 512
            signal = np.pad(signal, (0, 512 - signal.shape[0] % 512))
            frames = signal.reshape(-1, 512)
            for frame in frames:
                data.append(frame)
                labels.append(genre)

    # Convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Apply PCA to the data
    pca = PCA(n_components=20)
    data = pca.fit_transform(data)

    np.save('models/data.npy', data)
    np.save('models/labels.npy', labels)

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

    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # Predict the genres of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Load a song to predict its genre
    mfcc_features = scaler.transform(load_song(song_path, pca))

    # Predict the genre of the song
