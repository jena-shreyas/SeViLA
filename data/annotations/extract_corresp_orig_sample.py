import json

# Load the annotations
with open('mixtral/data_annotated.json', 'r') as f:
    annot_data = json.load(f)   # Dict

# Load the original train.json file from which the annotation samples were extracted
with open('original/train.json', 'r') as f:
    orig_data = json.load(f)    # List

annot_qids = list(annot_data.keys())

orig_data_sample = {}

for d in orig_data:
    if d['qid'] in annot_qids:
        orig_data_sample[d['qid']] = d

assert len(orig_data_sample) == len(annot_qids)

with open('mixtral/data_orig_sample.json', 'w') as f:
    json.dump(orig_data_sample, f)
