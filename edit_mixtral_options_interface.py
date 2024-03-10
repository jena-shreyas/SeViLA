import os
import gradio as gr
import json
import random
import sys

random.seed(42)

video_dir = 'data/videos'
annot_dir = 'data/annotations/mixtral'
filename = "train"
annot_path = os.path.join(annot_dir, filename + '.json')
save_path = os.path.join(annot_dir, filename + "_filtered.json")
num_options = 5
num_samples = 10

# Load the annotations
with open(annot_path, 'r') as f:
    data = json.load(f)

try:
    with open(save_path, 'r') as f:
        filtered_data = json.load(f)
except:
    filtered_data = {}
    filtered_data['start_idx'] = 0

start_idx = filtered_data['start_idx']

def option_demo(video, question, question_type, question_id, option1, option2, option3, option4, option5, answer):
    global filtered_data, num_options, start_idx
    video = video.split('/')[-1].split('.')[0]
    filtered_data[video] = {
        "video": video,
        "question": question,
        "qn_type": question_type,
        "qid": question_id,
        "num_option": num_options,
        "a0": option1,
        "a1": option2,
        "a2": option3,
        "a3": option4,
        "a4": option5,
        "answer": answer
    }
    filtered_data['start_idx'] += 1
    return option1, option2, option3, option4, option5

try:
    with gr.Blocks(title="Mixtral-7B option editing") as demo:
        description = """<p style="text-align: center; font-weight: bold;">
            <span style="font-size: 28px">Mixtral-Instruct-7B-v2 Augmented Options Editing Interface</span>
        </p>
        <p>
            Instructions:
            <br>
            (1) Edit the options in-place in case you want to change them. 
            <br>
            (2) Click on the "Edit" button to save changes.
            <br>
        </p>
        """
        gr.HTML(description)
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                video = gr.Video(label='Video') 
                question = gr.Textbox(placeholder="Why did the two ladies put their hands above their eyes while staring out?", label='Question')
                question_type = gr.Textbox(placeholder="counterfactual", label='Question Type')
                question_id = gr.Textbox(placeholder="question_id", label='Question ID')
                with gr.Row():
                    option1 = gr.Textbox(placeholder="option 1", label='Option 1')
                with gr.Row():
                    option2 = gr.Textbox(placeholder="option 2", label='Option 2')
                with gr.Row():
                    option3 = gr.Textbox(placeholder="option 3", label='Option 3')
                with gr.Row():
                    option4 = gr.Textbox(placeholder="option 4", label='Option 4')
                with gr.Row():        
                    option5 = gr.Textbox(placeholder="option 5", label='Option 5')
                with gr.Row():
                    answer = gr.Textbox(placeholder="answer", label='Answer')                
           
                gen_btn = gr.Button(value='Edit options')
            with gr.Column(scale=1, min_width=600): 
                
                option1_ = gr.Textbox(label='Option 0')
                option2_ = gr.Textbox(label="Option 1")
                option3_ = gr.Textbox(label="Option 2")
                option4_ = gr.Textbox(label="Option 3")
                option5_ = gr.Textbox(label="Option 4")
            
            gen_btn.click(
                option_demo,
                inputs=[video, question, question_type, question_id, option1, option2, option3, option4, option5, answer],
                outputs=[option1_, option2_, option3_, option4_, option5_],
                queue=True
            )
        with gr.Column():
            data_examples = []
            for qn_dict in data:
                vid = os.path.join(video_dir, qn_dict['video'] + '.mp4')
                qn = qn_dict['question']
                qn_type = qn_dict['qn_type']
                qid = qn_dict['qid']
                a0 = qn_dict['a0']
                a1 = qn_dict['a1']
                a2 = qn_dict['a2']
                a3 = qn_dict['a3']
                a4 = qn_dict['a4']
                options = [a0, a1, a2, a3, a4]
                ans = "Option " + str(qn_dict['answer'])
                data_examples.append([vid, qn, qn_type, qid, *options, ans])

            random.shuffle(data_examples)
            data_samples = data_examples[start_idx:start_idx+num_samples]

            gr.Examples(
                inputs=[video, question, question_type, question_id, option1, option2, option3, option4, option5, answer],
                outputs=[option1_, option2_, option3_, option4_, option5_],
                fn=option_demo,
                examples=data_samples,
                cache_examples=False,
            )

        demo.queue(api_open=False)          
        demo.launch(share=False) 

# Click Ctrl+C TWICE to save the filtered data and exit
except KeyboardInterrupt:
    with open(save_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"{filtered_data['start_idx']} questions edited.")
    print("Saved filtered data to", save_path)
    sys.exit(0)