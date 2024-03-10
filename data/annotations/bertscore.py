from evaluate import load
import json
from tqdm import tqdm

# For each annotated counterfactual option, we wish to compute its relevance to the question and the correct option.
# So, for the options :
# - correct option : compute BERTScore w.r.t question only
# - incorrect option : some fn of BERTscore w.r.t. question and correct option
# Since BERTScore is a set of precision, recall and f1, we use BERTScore F1

bertscore = load("bertscore")

with open('mixtral/data_orig_sample.json', 'r') as f:
    orig_data =json.load(f)

with open('mixtral/data_annotated.json', 'r') as f:
    annot_data = json.load(f)


# def f(wr_qn_bertscore, wr_cr_op_bertscore):
#     # higher weight to question since wrong option should always be somewhat relevant to question,
#     # but can still be a bit relevant to correct option
#     # return 0.6 * wr_qn_bertscore  + 0.4 * wr_cr_op_bertscore
#     return wr_cr_op_bertscore

def compute_bertscore(data: dict):
    avg_cr_qn_bertscore = 0
    avg_wr_qn_bertscore = 0
    avg_wr_cr_op_bertscore = 0
    n = len(data)
    for _, d in tqdm(data.items(), total=n):
        qn = d['question']
        num_options = d['num_option']
        options = [d[f'a{i}'] for i in range(num_options)]
        correct_option = options[int(d['answer'])]

        # compute BERTscore b/w qn and correct option
        results = bertscore.compute(predictions=[correct_option], references=[qn], model_type="distilbert-base-uncased")
        avg_cr_qn_bertscore += results["f1"][0]/n

        # compute BERTscore for incorrect options
        for i, option in enumerate(options):
            if option != correct_option:
                # compute BERTscore b/w qn and option
                result1 = bertscore.compute(predictions=[option], references=[qn], model_type="distilbert-base-uncased")
                wr_qn_bertscore = result1["f1"][0]
                avg_wr_qn_bertscore += wr_qn_bertscore/((num_options - 1) * n)

                # compute BERTscore b/w correct option and option
                result2 = bertscore.compute(predictions=[option], references=[correct_option], model_type="distilbert-base-uncased")
                wr_cr_op_bertscore = result2["f1"][0]
                avg_wr_cr_op_bertscore += wr_cr_op_bertscore/((num_options - 1) * n)


    return avg_cr_qn_bertscore, avg_wr_qn_bertscore, avg_wr_cr_op_bertscore


corr_orig, wr_qn_orig, wr_cr_op_orig = compute_bertscore(orig_data)
corr_annot, wr_qn_annot, wr_cr_op_annot = compute_bertscore(annot_data)

print(f'Original data: correct option-question bertscore: {corr_orig}, wrong option-question bertscore: {wr_qn_orig}, wrong option-correct option bertscore: {wr_cr_op_orig}')
print(f'Annotated data: correct option bertscore: {corr_annot}, wrong option-question bertscore: {wr_qn_annot}, wrong option-correct option bertscore: {wr_cr_op_annot}')

