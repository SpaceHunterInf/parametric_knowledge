import json
import torch

if __name__ == '__main__':
    filename = 'ukp_test/bertnli_base_.json'

    with open(filename, 'r') as f:
        data = json.load(f)

    print(len(data))
    total = len(data)
    both_agree = 0
    base_acc = 0
    W_acc = 0
    Aw_acc = 0
    for d in data:
        if d['WarrantBelief_base'][0] > d['WarrantBelief_base'][1]:
            if d['AWarrantBelief_base'][0] > d['AWarrantBelief_base'][1]:
                both_agree += 1
        
        if torch.tensor(d['NoWarrant_base']).argmax(-1).item() == 0:
            base_acc +=1
        if torch.tensor(d['Warrant_base']).argmax(-1).item() == 0:
            W_acc +=1
        if torch.tensor(d['AWarrant_base']).argmax(-1).item() == 2:
            Aw_acc +=1
        
    print(both_agree/total)
    print(base_acc/total)
    print(W_acc/total)
    print(Aw_acc/total)