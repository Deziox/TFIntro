import librosa

y,sr = librosa.load(librosa.util.example_audio_file())
print(librosa.feature.melspectrogram(y,sr))
