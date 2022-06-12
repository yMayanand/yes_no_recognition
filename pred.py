import torch
from model import Model

model = Model()
state_dict = torch.load('checkpoint0.pt')
model.load_state_dict(state_dict['model_state'])

idx2val = {0: 'NO', 1: 'YES', 2: 'EPSILON'}


def predict(data):
    model.eval
    with torch.no_grad():
        out = model(data)


    preds = torch.argmax(out.squeeze(1), dim=1)
    results = []
    
    for i in preds:
        v = idx2val[i.item()]
        results.append(v)
    
    print(results)
    
    refined = []
    count = 0
    for i in results:
        if not refined:
            if i != 'EPSILON':
                refined.append(i)
        elif i != 'EPSILON' and results[count - 1] != i:
            refined.append(i)
        count += 1
        



    return ' '.join(refined)

