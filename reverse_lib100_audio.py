'''
import sys
sys.path.append('/users/PAS2065/yang4581/.conda/envs/tts/lib/python3.9/site-packages')

# reverse the audio
from pydub import AudioSegment
from pydub.playback import play

def reverse_and_export_flac_audio(path: str) -> None:
  song = AudioSegment.from_file(path, "flac")
  backwards = song.reverse()
  backwards.export(path, format="flac")

import os, fnmatch
for root, dirnames, filenames in os.walk("/fs/scratch/PAS2400/TTS_baseline/training_data/LibriSpeech/train-clean-100/"):
    for filename in filenames:
        if filename.endswith("flac"):
            reverse_and_export_flac_audio(os.path.join(root, filename))
'''


import pdb
import torch
import torchaudio

def reverse_and_export_flac_audio(path: str) -> None:
    audio, sr = torchaudio.load(path)
    reverse_audio = torch.flip(audio,(0,1))
    #pdb.set_trace()
    rpath = path.replace("train-clean-100", "train-clean-100_reversed")
    torchaudio.save(rpath, reverse_audio, sr)

import os, fnmatch
for root, dirnames, filenames in os.walk("/fs/scratch/PAS2400/TTS_baseline/training_data/LibriSpeech/train-clean-100/"):
    for filename in filenames:
        if filename.endswith("flac"):
            #pdb.set_trace()
            isExist = os.path.exists(root.replace("train-clean-100", "train-clean-100_reversed"))
            if not isExist:
                os.makedirs(root.replace("train-clean-100", "train-clean-100_reversed"))
            reverse_and_export_flac_audio(os.path.join(root, filename))