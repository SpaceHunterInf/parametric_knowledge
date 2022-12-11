import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os

random.seed(557)
id2label = {'0':'entailment', '1':'neutral', '2':'contradiction'}
label2id = {'entailment':0, 'neutral':1, 'contradiction':2}

class NLIDataset(Dataset):    
    def __init__(self, args, data, tokenizer):          
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
     
    def __getitem__(self, idx):
        x = {
            'sentence1' : self.data[idx]['sentence1'],
            'sentence2' : self.data[idx]['sentence2'],
            'label' : self.data[idx]['gold_label']
        }
        if 'bert' in self.args['model_name']:
            input_ids, attention_mask, token_type_ids = bert_preprocess(self.tokenizer, x, self.args['max_len'])
            # DONE!
            label = label2id[x['label']]
            #label = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=len(label2id)).to(torch.float)
            return {
                    "label": label,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                    }

    def __len__(self):
        return len(self.data)

def bert_preprocess(tokenizer, input_dict, length):
    inputs = tokenizer.encode_plus(input_dict['sentence1'], input_dict['sentence2'], add_special_tokens=True, max_length=length)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    # BERT requires sequences in the same batch to have same length, so let's pad!
    padding_length = length - len(input_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Convert them into PyTorch format.
    #label = torch.tensor(int(x["quality"])).long()
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    return input_ids, attention_mask, token_type_ids

def prepare_data(args, tokenizer):
    if args['data'] == 'SNLI':
        df_train = pd.read_csv("snli_data/snli_1.0_train.txt", sep="\t")
        df_dev = pd.read_csv("snli_data/snli_1.0_dev.txt", sep="\t")
        df_test = pd.read_csv("snli_data/snli_1.0_test.txt", sep="\t")
    
        df_train = df_train[['gold_label','sentence1','sentence2']].to_dict(orient='records')
        df_train = [x for x in df_train if x['gold_label'] in ['entailment', 'neutral', 'contradiction']]
        df_dev = df_dev[['gold_label','sentence1','sentence2']].to_dict(orient='records')
        df_dev = [x for x in df_dev if x['gold_label'] in ['entailment', 'neutral', 'contradiction']]
        df_test = df_test[['gold_label','sentence1','sentence2']].to_dict(orient='records')
        df_test = [x for x in df_test if x['gold_label'] in ['entailment', 'neutral', 'contradiction']]
        
    #print(df_test)
    train_dataset = NLIDataset(args, df_train, tokenizer)
    dev_dataset = NLIDataset(args, df_dev, tokenizer)
    test_dataset = NLIDataset(args, df_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, num_workers=16)
    
    return train_loader, dev_loader, test_loader
        


        