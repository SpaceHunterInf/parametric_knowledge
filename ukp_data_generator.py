import json

if __name__ == '__main__':
    names = ['train','dev', 'test']
    for name in names:

        with open('ukp_data/filtered_{}.json'.format(name),'r') as f:
            data = json.load(f)

        ukp_finetune = []
        for d in data:
            premise = d['reason']
            hypothesis = d['claim']

            if d['correctLabelW0orW1'] == 0:
                warrant = 'warrant0'
                antiwarrant = 'warrant1'
            else:
                warrant = 'warrant1'
                antiwarrant = 'warrant0'

            correct_premise = premise + 'Given that ' + d[warrant]
            wrong_premise = premise + 'Given that ' + d[antiwarrant]

            correct_example = {'sentence1': correct_premise, 'sentence2': hypothesis, 'gold_label':'entailment'}
            wrong_example = {'sentence1': wrong_premise, 'sentence2': hypothesis, 'gold_label':'contradiction'}

            ukp_finetune.append(correct_example)
            ukp_finetune.append(wrong_example)

        with open('ukp_data/ukp_{}.json'.format(name),'w') as f:
            f.write(json.dumps(ukp_finetune, indent=2))
            f.close()