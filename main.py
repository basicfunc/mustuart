# Step 1: Import required libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D,\
    MaxPooling1D, BatchNormalization, Activation, Flatten


# Step 2: Load and preprocess the GTZAN dataset
def load_gtzan_data(gtzan_dir, num_segments=5, n_mfcc=13,
                    n_fft=2048, hop_length=512):
    genres = os.listdir(gtzan_dir)
    data = []
    labels = []

    for i, genre in enumerate(genres):
        genre_dir = os.path.join(gtzan_dir, genre)
        for filename in os.listdir(genre_dir):
            song_path = os.path.join(genre_dir, filename)
            samples, sample_rate = librosa.load(song_path, sr=None)
            duration = samples.shape[0] // sample_rate

            for j in range(num_segments):
                start = j * duration // num_segments * sample_rate
                end = (j + 1) * duration // num_segments * sample_rate
                mfcc = librosa.feature.mfcc(
                    y=samples[start:end], sr=sample_rate, n_mfcc=n_mfcc,
                    n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T
                if mfcc.shape[0] < n_mfcc:
                    # Pad with zeros
                    mfcc = np.pad(mfcc, ((0, n_mfcc - mfcc.shape[0]), (0, 0)))
                data.append(mfcc)
                labels.append(i)

    return np.array(data), np.array(labels), genres


# Set the path to the GTZAN dataset
gtzan_dir = "genres_original/"

data, labels, genres = load_gtzan_data(gtzan_dir)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42)

print(X_train)
print(X_test)

# Step 3: Define a neural network model
def create_model(input_shape):
    model = Sequential()

    model.add(Conv1D(64, kernel_size=3, input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))

    model.add(Conv1D(128, kernel_size=3))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3))

    model.add(LSTM(64))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = create_model(input_shape)

# Step 4: Train the model
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), batch_size=32, epochs=50)


# Step 5: Create a function to load and preprocess a new song
def preprocess_song(song_path, n_mfcc=13, n_fft=2048, hop_length=512):
    samples, sample_rate = librosa.load(song_path, sr=None)
    mfcc = librosa.feature.mfcc(
        samples, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length)
    mfcc = mfcc.T
    return np.array([mfcc])


# Step 6: Get user input and predict genre
song_path = input("Enter the path to the song: ")
song_data = preprocess_song(song_path)
prediction = model.predict(song_data)
predicted_genre = genres[np.argmax(prediction)]
print(f"The predicted genre for the song is: {predicted_genre}")
