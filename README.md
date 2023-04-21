# Mustuart
This project aims to build a music genre classification engine using machine learning models trained on the GTZAN dataset.

# Dataset
The GTZAN dataset contains 1000 30-second audio clips evenly split between 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock). The clips are all 22050 Hz Mono 16-bit audio files in .wav format.
The GTZAN dataset is limited to only English songs, which may not represent the diversity of music across different cultures and languages. Additionally, the dataset is small, and some of the songs are not of high quality, which may affect the performance of the models trained on the dataset.

Datset can be found here - https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification/download

# Features
The audio clips are converted into feature vectors using `librosa` and `python_speech_features`. Features extracted include:

- MFCC (Mel Frequency Cepstral Coefficients)
- Chroma Frequencies
- Mel Spectrogram
- Contrast

These feature vectors are then used to train machine learning models.

# Models
Various models are trained on the extracted MFCC features to classify and predict the genre of the songs. The models trained are CNN, CNN with RNN, SVM with PCA, SVM with PCA and GridsearchCV, and simple SVM.
The best performing SVM model was embedded into a Python script (~700KB) and can be found in the `saved_model` folder.
A Flask app in the `app/` directory provides an interface to the classification engine. The `recommend.py` uses the AudD API to provide music recommendations based on the embedded SVM model.
The recommendation engine does not currently work fully due to limitations with the AudD API.

# Future Work
- Use a dataset with a wider range of genres like FMA
- Fine tune the machine learning models
- Deploy the app with WASM to avoid browser limitations
- Explore a larger choice of models like neural networks

# FMA dataset
The FMA dataset is a large, diverse dataset of freely licensed music. The dataset contains 106,574 tracks from 16,341 artists and 14,854 albums, covering a wide range of genres and sub-genres. The FMA dataset is a good alternative to the GTZAN dataset, as it provides a larger and more diverse set of music for training machine learning models.

# Conclusion:
Mustuart is a music genre classification engine that utilizes machine learning models trained on the GTZAN dataset to classify songs based on their genre. The project uses various libraries such as `librosa`, `python_speech_features`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, etc. to extract features, train models and process data. The trained SVM model is embedded into a Flask app for easy deployment and usage. While the GTZAN dataset is limited to only English songs, the FMA dataset provides a larger and more diverse set of music for training machine learning models.
