# EmotionAnalyzerFromYoutubeAudio
This repository allows you to retrieve emotion in two classes: Arousal and Valence. The youtube audio is downloaded and emotion analysis is done on full spectrum, foreground  and background.

## Requirements
To work with this repository, you need to have following dependencies installed on python 2.7

pip install youtube_dl

pip install librosa

pip install pyAudioAnalysis

once the pyAudioAnalysis is installed, run the following command:

#### python audioAnalysis.py trainRegression -i data/speechEmotion/ --method svm -o data/svmSpeechEmotion 

Make sure to make changes in model path and audio slice dir in jupyter notebook before running.
https://github.com/babupriyavrat/EmotionAnalyzerFromYoutubeAudio/blob/master/Emotion%20Detection%20from%20Youtube%20Audio.ipynb

## Parallel processing

In order to speed up wav file generation, you can run wave_chunk_generator.py based on the number of CPUs you want to assign to the wav file generation.

To get emotion clasess after generating wav files, run Emotion_extractor.py. Parallel processing needs to be implemented to improve performance. A 110 minute audio tooks 19 minutes to analyze emotionally.


