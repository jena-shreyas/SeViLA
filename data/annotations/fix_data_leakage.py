import json
import random

random.seed(42)

with open("original/train_old.json", 'r') as f:
    train = json.load(f)

with open("original/test_old.json", 'r') as f:
    test = json.load(f)

with open("mixtral/orig_sample.json", 'r') as f:
    sample = json.load(f)

test_sample = random.sample(test, len(sample))

train_rem = [qn_d for qn_d in train if qn_d not in sample]
assert len(train_rem) == len(train) - len(sample)   # sample is a subset of train

train_new = train_rem + test_sample
random.shuffle(train_new)      # take a 300-qn sample from val set and add to train set
test_new = [qn_d for qn_d in test if qn_d not in test_sample] + sample
random.shuffle(test_new)             # remove the 300-qn sample from val set

assert len(train_new) == len(train)
assert len([qn_d for qn_d in train_new if qn_d in sample]) == 0
assert len(test_new) == len(test)

with open("original/train.json", 'w') as f:
    json.dump(train_new, f)

with open("original/test.json", 'w') as f:
    json.dump(test_new, f)


