import sys
from audio import download_audio, detect_audio

if __name__ == '__main__':
    _, _id = sys.argv
    url = 'https://www.youtube.com/watch?v=' + _id
    print(url)
    download_audio(url, _id)
    detect_audio(_id)