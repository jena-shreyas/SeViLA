'''
    Convert CausalVidQA data split files from csv to json format for SeViLA.
'''


import os
import sys
import pandas as pd
import json
from tqdm import tqdm
import shutil

def setup_data(vid_path: str, csv_path: str, 
                save_path: str, split: str):

    for video_id in tqdm(os.listdir(vid_path)):
        try:
            shutil.copy(os.path.join(vid_path, video_id, video_id + '.mp4'), os.path.join(save_path, 'videos', video_id + '.mp4'))
        except FileNotFoundError:
            pass

    # data = []

    # df = pd.read_csv(csv_path)
    # for i in tqdm(range(len(df))):
    #     row = df.iloc[i]
    #     question = row['question']
    #     qn_type = row['qn_type']
    #     video_id = row['video_id']
    #     a0 = row['a0']
    #     a1 = row['a1']
    #     a2 = row['a2']
    #     a3 = row['a3']
    #     a4 = row['a4']
    #     answer = row['answer']

    #     data_dict = {}
    #     data_dict['question'] = question
    #     data_dict['num_option'] = 5
    #     data_dict['a0'] = a0
    #     data_dict['a1'] = a1
    #     data_dict['a2'] = a2
    #     data_dict['a3'] = a3
    #     data_dict['a4'] = a4
    #     data_dict['answer'] = int(answer)
    #     data_dict['video'] = video_id
    #     data_dict['qid'] = video_id + "_" + qn_type
    #     data_dict['qn_type'] = qn_type

    #     data.append(data_dict)

    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, "annotations", f"{split}.json"), "w") as f:
    #     json.dump(data, f)


def main(args):
    vid_path = args[1]
    csv_path = args[2]
    save_path = args[3]
    split = csv_path.split('/')[-1].split('.')[0]

    setup_data(vid_path, csv_path, save_path, split)

if __name__ == '__main__':
    main(sys.argv)
