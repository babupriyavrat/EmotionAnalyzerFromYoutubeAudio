import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
from functools import partial

def load_audio_slice(file_name,offset_in_sec,period_in_sec):
    y, sr = librosa.load(file_name,  
                       duration=period_in_sec,offset=offset_in_sec)
    return y,sr

def computeAbsMagandPhase(y,sr):
    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))
    S=np.abs(librosa.stft(y))
    
    return S,S_full,phase

def SeparateVoiceAndBackground(S_full,phase,sr):
    ###########################################################
    # The wiggly lines above are due to the vocal component.
    # Our goal is to separate them from the accompanying
    # instrumentation.
    #

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.
    S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)
    ##############################################
    # The raw filter output can be used as a mask,
    # but it sounds better if we use soft-masking.

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    return S_foreground, S_background


def audio_convertor(input_file,chunk_number,file_type,S_Type,phase,sr):
    librosa.output.write_wav(str(input_file)+"_"+str(chunk_number)+"_"+str(file_type)+".wav", librosa.istft(S_Type* phase), sr)
    

def create_durations(execution_id):
    interval=10
    file_name=str(execution_id)
    slice_audio_dir="audio_tmp"
    model_path="model.svm"
    y, sr = librosa.load(file_name+".wav")
    duration= librosa.get_duration(y=y, sr=sr)
    print("The duration of the file :"+file_name+ " is "+str(duration))
    duration_set=list(range(0,int(duration - interval),interval/2))
    return duration,duration_set

def create_wav_chunks(from_duration,execution_id):
    interval=10
    file_name = str(execution_id)
    to_duration=from_duration+interval
    chunk=str(from_duration)+"to"+str(to_duration)+"sec"
    y, sr=load_audio_slice(file_name+".wav",from_duration,interval)
    S,S_full,phase=computeAbsMagandPhase(y,sr)
    S_foreground,S_background=SeparateVoiceAndBackground(S_full,phase,sr)
    audio_convertor(file_name[0:15],chunk,"S_full",S_full,phase,sr)
    audio_convertor(file_name[0:15],chunk,"S_foreground",S_foreground,phase,sr)
    audio_convertor(file_name[0:15],chunk,"S_background",S_background,phase,sr)