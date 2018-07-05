import youtube_dl
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from functools import partial

from emotion_analyzer import EmotionAnalyzerSVM
from wavechunk_generator import create_durations, create_wav_chunks

def download_audio(link, execution_id):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(execution_id) + '.wav'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

def detect_audio(execution_id):
    duration, duration_set = create_durations(execution_id)
    
    pool = ThreadPool(4) 
    
    create_wav_chunks_partial=lambda x: create_wav_chunks(x,execution_id)
    
    pool.map(create_wav_chunks_partial,duration_set) # blocking
    pool.close() 
    pool.join()
    
    EmotionAnalyzerSVM(duration, execution_id)