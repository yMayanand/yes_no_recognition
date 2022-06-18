import torch
import numpy as np
import torchaudio
from torchvision import transforms
from pred import predict
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(duration, sr, c=1):
    print('recording started ...')
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=c)
    
    # Record audio for the given number of seconds
    sd.wait()
    print('recording stopped')
    return recording

tfms = tfms = transforms.Compose([
    torchaudio.transforms.Spectrogram(),
    torchaudio.transforms.AmplitudeToDB(),
    transforms.Resize((201, 256))
])

data = record_audio(6, 8000, c=2)
data = data.T[0]
data = torch.tensor(np.expand_dims(data, 0))

print(predict(tfms(data).unsqueeze(0)))
