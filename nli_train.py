import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from copy import deepcopy
import json
from tqdm import tqdm
from config import *
from data_loader import *
import random, os

class nli_task(pl.LightningModule):
    
    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]
        self.args = args


    def training_step(self, batch, batch_idx):
        self.model.train()
        if 'bert' in self.args['model_name']:
            #print(batch)
            # result = pl.TrainResult(loss)
            # result.log('train_loss', loss, on_epoch=True)
            loss = self.model(batch["input_ids"],attention_mask = batch["attention_mask"],token_type_ids = batch["token_type_ids"], labels=batch['label']).loss
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if 'bert' in self.args['model_name']:
            #print(batch)
            # result = pl.TrainResult(loss)
            # result.log('train_loss', loss, on_epoch=True)
            #print((len(batch["input_ids"]), len(batch["attention_mask"]), len(batch["token_type_ids"]), len(labels)))
            loss = self.model(batch["input_ids"],attention_mask = batch["attention_mask"],token_type_ids = batch["token_type_ids"], labels=batch['label']).loss
        print(loss)
        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()}, 'val_loss': val_loss_mean.item()}
        self.log("val_loss", results['val_loss'])
        return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

def train(args, *more):
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"] + str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    seed_everything(args["seed"])


    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bert" in args["model_name"]:
        model = BertForSequenceClassification.from_pretrained(args["model_checkpoint"], num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(args["model_checkpoint"])
        model.config.id2label = {'0':'entailment', '1':'neutral', '2':'contradiction'}
        model.config.label2id = {'entailment':0, 'neutral':1, 'contradiction':2}

    task = nli_task(args, tokenizer, model)

    train_loader, val_loader, test_loader = prepare_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    #callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    #precision=16,
                    accelerator="cuda"
                    )

    trainer.fit(task, train_loader, val_loader)

    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    evaluate_model(args, task.tokenizer, task.model, test_loader, save_path)

def evaluate_model(args, tokenizer, model, test_loader, save_path):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred_positive = {'entailment':1, 'neutral':1, 'contradiction':1}
    gold_positive = {'entailment':1, 'neutral':1, 'contradiction':1}
    pred_tp = {'entailment':1, 'neutral':1, 'contradiction':1}
    
    save = []

    for batch in tqdm(test_loader):
        with torch.no_grad():
            #print(batch)
            logits = model(batch["input_ids"],attention_mask = batch["attention_mask"],token_type_ids = batch["token_type_ids"]).logits
            predicted_class_ids = torch.argmax(logits, dim=1)
            #print(predicted_class_ids)
            for idx in range(len(predicted_class_ids)):
                pred_label = model.config.id2label[str(predicted_class_ids[idx].item())]
                pred_positive[pred_label] += 1
                gold_label = model.config.id2label[str(batch['label'][idx].item())]
                gold_positive[gold_label] +=1
                if pred_label == gold_label:
                    pred_tp[gold_label] += 1

                tmp_save = {'sentence1': batch['sentence1'][idx], 'sentence2': batch['sentence2'][idx], 'gold_label':gold_label}
                tmp_save['pred_label'] = pred_label
        
                save.append(tmp_save)
    
    class_f1 = {}

    for key in pred_positive.keys():
        pre = pred_tp[key] / pred_positive[key]
        rec = pred_tp[key] / gold_positive[key]
        class_f1[key] = (2* pre * rec) / (pre + rec)
    
    save += [pred_positive, gold_positive, pred_tp, class_f1]

    total_tp = 1
    for key in pred_tp.keys():
        total_tp += pred_tp[key]

    total = 1
    for key in gold_positive.keys():
        total += gold_positive[key]

    print('Total Acc: {}'.format(str(total_tp/total)))

    with open(os.path.join(save_path,'results.json'), 'w') as f:
        f.write(json.dumps(save, indent=2))
        f.close

if __name__ == '__main__':
    args = get_args()
    if args.mode=="train":
        train(args)