import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT

import csv
def EmotionAnalyzerSVM(duration, execution_id):
    slice_audio_dir = 'audio_tmp'
    model_path = 'model.svm'

    interval = 10
    file_name = str(execution_id)
    with open(file_name + '_audio.csv','w') as outfile: 
        columns=[
            "from",
            "to",
            "next_offset",
            "Full_Arousal",
            "Full_Valence",
            "Foreground_Arousal",
            "Foreground_Valence",
            "Background_Arousal",
            "Background_Valence"
        ]
        w = csv.DictWriter(outfile,fieldnames=columns)
        w.writeheader()
        for i in range(0,int(duration),interval/2):
            from_duration=i
            to_duration=interval+i
            next_offset=i+interval/2
            chunk=str(from_duration)+"to"+str(to_duration)+"sec"
            outfile.write(str(from_duration)+','
                              +str(to_duration)+','
                              +str(next_offset)+',')
            try:
                sentiment=aT.fileRegression(slice_audio_dir+file_name[0:15]+"_"+chunk+"_"+"S_full"+".wav",model_path, "svm")[0]
                #print("sentiment for Full Spectrum"+chunk+"- Arousal:"+sentiment[0]+"; Valence:"+sentiment[1] )
                outfile.write(str(sentiment[0])+','
                              +str(sentiment[1])+',')
            except TypeError:
                pass
            try:
                sentiment2=aT.fileRegression(slice_audio_dir+file_name[0:15]+"_"+chunk+"_"+"S_foreground"+".wav", model_path, "svm")[0]
                #print("sentiment for Foreground"+chunk+"- Arousal:"+sentiment2[0]+"; Valence:"+sentiment2[1] )
                outfile.write(
                             str(sentiment2[0])+','
                              +str(sentiment2[1])+',')
            except TypeError:
                pass
            try:
                sentiment3=aT.fileRegression(slice_audio_dir+file_name[0:15]+"_"+chunk+"_"+"S_background"+".wav", model_path, "svm")[0]
                #print("sentiment for background"+chunk+"- Arousal:"+sentiment3[0]+"; Valence:"+sentiment3[1] )
                outfile.write(
                             str(sentiment3[0])+','
                             +str(sentiment3[1]))
            except TypeError:
                pass
            
    	    outfile.write("\n")
            
# EmotionAnalyzerSVM('6495',str(slice_audio_dir),str(model_path),execution_id)