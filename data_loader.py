import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import json
from functools import partial
from transformers import  DataCollatorForSeq2Seq

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
        if 'bert' in self.args['model_name'] and not 'roberta' in self.args['model_name']:
            input_ids, attention_mask, token_type_ids = bert_preprocess(self.tokenizer, x, self.args['max_len'])
            # DONE!
            label = label2id[x['label']]
            #label = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=len(label2id)).to(torch.float)
            return {
                    "label": label,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "sentence1": x['sentence1'],
                    "sentence2": x['sentence2']
                    }
        elif 'roberta' in self.args['model_name']:
            input_ids, attention_mask = roberta_preprocess(self.tokenizer, x, self.args['max_len'])
            label = label2id[x['label']]
            return {'label':label, 'input_ids':input_ids, 'attention_mask':attention_mask,"sentence1": x['sentence1'],"sentence2": x['sentence2']}            
        elif 't5' in self.args['model_name']:
            input_text = "Infer the following 2 sentences: " +  'Premise: ' + x['sentence1'] + ' Hypothesis: ' + x['sentence2']
            output_text = x['label']
            model_input = {'input_text':input_text}
            model_input['output_text'] = output_text
            model_input['sentence1'] = x['sentence1']
            model_input['sentence2'] = x['sentence2']
            model_input['label'] = x['label']
            return model_input

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

def roberta_preprocess(tokenizer, input_dict, length):
    inputs = tokenizer(input_dict['sentence1'], input_dict['sentence2'])
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    
    padding_length = length - len(input_ids)
    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    
    input_ids = torch.tensor(input_ids)#.unsqueeze(0)
    attention_mask = torch.tensor(attention_mask)#.unsqueeze(0)
    return input_ids, attention_mask
    


def generate_ukp_input(inputs, prompt, mode):
    generated_inputs = []
    for data in inputs:
        input_dict = {'sentence1':data['reason'], 'sentence2':data['claim'], 'gold_label':'entailment'}
        if mode == 'base':
            pass
        elif mode == 'W':
            if data['correctLabelW0orW1'] == 0:
                warrant = data['warrant0']
            else:
                warrant = data['warrant1']
            input_dict['sentence1'] += prompt + warrant
        elif mode == 'AW':
            if data['correctLabelW0orW1'] == 0:
                antiwarrant = data['warrant1']
            else:
                antiwarrant = data['warrant0']
            input_dict['sentence1'] += prompt + antiwarrant
        generated_inputs.append(input_dict)
    return generated_inputs
        

def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["input_ids"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["label_ids"] = output_batch['input_ids']

    return batch_data

def prepare_data(args, tokenizer):
    if args['data'] == 'SNLI':
        with open('snli_data/filtered_train.txt', 'r') as f:
            df_train = json.load(f)
        with open('snli_data/filtered_dev.txt', 'r') as f:
            df_dev = json.load(f)
        with open('snli_data/filtered_test.txt', 'r') as f:
            df_test = json.load(f)
    elif args['data'] == 'UKP':
        with open('ukp_data/filtered_train.json', 'r') as f:
            df_train = json.load(f)
        with open('ukp_data/filtered_dev.json', 'r') as f:
            df_dev = json.load(f)
        with open('ukp_data/filtered_test.json', 'r') as f:
            df_test = json.load(f)
        
        df_train = generate_ukp_input(df_train, args['ukp_prompt'], args['ukp_mode'])
        df_dev = generate_ukp_input(df_dev, args['ukp_prompt'], args['ukp_mode'])
        df_test = generate_ukp_input(df_test, args['ukp_prompt'], args['ukp_mode'])

        total = df_train + df_dev + df_test
    #print(df_test)
    train_dataset = NLIDataset(args, df_train, tokenizer)
    dev_dataset = NLIDataset(args, df_dev, tokenizer)
    test_dataset = NLIDataset(args, df_test, tokenizer)
    
    if 't5' in args['model_name']:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, num_workers=16)

    
    return train_loader, dev_loader, test_loader
        


        