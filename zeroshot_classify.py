from transformers import pipeline
import json
from tqdm import tqdm

if __name__ == '__main__':

    model_path = 'save/bert-base-uncasedbert0.0001_epoch_5_seed_557'
    classifier = pipeline("zero-shot-classification", model = model_path, device=0)
    prompt = ""
    #initial knowledge probing
    classes = ['True', 'False']
    nli_labels = ['Entailment', 'Neutral', 'Contradiction']

    with open('ukp_data/filtered_train.json','r') as f:
        data = json.load(f)

    for d in tqdm(data, desc = 'testing'):
        if d['correctLabelW0orW1'] == 0:
            warrant = 'warrant0'
            antiwarrant = 'warrant1'
        else:
            warrant = 'warrant1'
            antiwarrant = 'warrant0'
        
        score = classifier(d[warrant], classes)['scores']
        d['WarrantBelief_base'] = score
        score = classifier(d[antiwarrant], classes)['scores']
        d['AWarrantBelief_base'] = score

        d['NoWarrant_base'] = classifier(d['reason']+d['claim'], nli_labels)['scores']
        d['Warrant_base'] = classifier(d['reason']+prompt+d[warrant]+d['claim'], nli_labels)['scores']
        d['AWarrant_base'] = classifier(d['reason']+prompt+d[antiwarrant]+d['claim'], nli_labels)['scores']


    with open('ukp_test/bertnli_base_'+prompt+'.json','w') as f:
        f.write(json.dumps(data, indent = 2))
        f.close()