from data import Dataset
from model import Model
import torch
import torch.nn as nn
from torchvision import transforms
import torchaudio
from tqdm import tqdm
import numpy as np

train_tfms = tfms = transforms.Compose([
    torchaudio.transforms.Spectrogram(),
    torchaudio.transforms.AmplitudeToDB(),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=50),
    torchaudio.transforms.TimeMasking(time_mask_param=10),
    transforms.Resize((201, 256))
])

val_tfms = tfms = transforms.Compose([
    torchaudio.transforms.Spectrogram(),
    torchaudio.transforms.AmplitudeToDB(),
    transforms.Resize((201, 256))
])

train_ds = Dataset(root='./', train_tfms=train_tfms, download=False, split=0.67)
val_ds = Dataset(root='./', val_tfms=val_tfms, train=False, download=False, split=0.67)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=40, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=20, drop_last=True)


S = 15     # Target sequence length of longest target in batch (padding length)
S_min = 8

def get_lengths(batch_size):
    input_lengths = torch.full(size=(batch_size,), fill_value=15,
                            dtype=torch.long)  # no of sequences in input
    # no if sequences in labels
    target_lengths = torch.full(size=(batch_size,), fill_value=8, dtype=torch.long)
    return input_lengths, target_lengths

ctc_loss = nn.CTCLoss(blank=2, zero_infinity=True)

model = Model()

def load_from_ckpt(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['model_state'])

load_from_ckpt(model, '/home/thinkin-machine/VS Code Workspaces/speech_recog/checkpoint1.pt')


for lr in [1e-5]:
    print(lr)
    print('*'*40)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if lr == 1e-3:
        epochs = 200
    else:
        epochs = 200

    for j in range(epochs):
        train_losses = []
        for i, batch in tqdm(enumerate(train_dl)):
            model.train()
            data, label = batch
            out = model(data)
            loss = ctc_loss(out, label, *get_lengths(40))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        val_losses = []
        
        for i, batch in tqdm(enumerate(val_dl)):
            model.eval()
            data, label = batch
            with torch.no_grad():
                out = model(data)
                loss = ctc_loss(out, label, *get_lengths(20))
            val_losses.append(loss.item())
        
        
            
                

        print(f'Epoch {j}: train_loss {np.mean(train_losses)}, val_loss {np.mean(val_losses)}')

state_dict = {'model_state': model.state_dict(),
              'optimizer_state': optimizer.state_dict()
              }

torch.save(state_dict, 'checkpoint1.pt')
