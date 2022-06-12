from data import Dataset
from model import Model
import torch
import torch.nn as nn
from torchvision import transforms
import torchaudio
from tqdm import tqdm

tfms = tfms = transforms.Compose([
    torchaudio.transforms.Spectrogram(),
    torchaudio.transforms.AmplitudeToDB(),
    transforms.Resize((201, 256))
])

train_ds = Dataset(root='./', tfms=tfms, download=False)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=30, drop_last=True)

S = 15     # Target sequence length of longest target in batch (padding length)
S_min = 8

input_lengths = torch.full(size=(30,), fill_value=15,
                           dtype=torch.long)  # no of sequences in input
# no if sequences in labels
target_lengths = torch.full(size=(30,), fill_value=8, dtype=torch.long)

ctc_loss = nn.CTCLoss(blank=2, zero_infinity=True)

model = Model()

for lr in [1e-3, 1e-4, 1e-5]:
    print(lr)
    print('*'*40)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if lr == 1e-3:
        epochs = 200
    else:
        epochs = 30

    for j in range(epochs):
        losses = []
        for i, batch in tqdm(enumerate(train_dl)):
            data, label = batch
            out = model(data)
            loss = ctc_loss(out, label, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f'Epoch {j}: loss {(losses[0] + losses[1])/2}')

state_dict = {'model_state': model.state_dict(),
              'optimizer_state': optimizer.state_dict()
              }

torch.save(state_dict, 'checkpoint0.pt')
