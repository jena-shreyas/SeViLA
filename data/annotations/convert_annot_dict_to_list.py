import json

with open('mixtral/data_orig_sample.json', 'r') as f:
    orig_data = json.load(f)

with open('mixtral/data_annotated.json', 'r') as f:
    annot_data = json.load(f)

orig_data_list = list(orig_data.values())
assert len(orig_data_list) == len(annot_data)

with open('mixtral/orig_sample.json', 'w') as f:
    json.dump(orig_data_list, f)