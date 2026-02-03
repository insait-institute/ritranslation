import gradio as gr
import json
import os
import warnings

warnings.filterwarnings("ignore")

# WIP: thanks o1 for helping me with the Gradio mess; this is adapted for MMLU so I will rewrite this later to be flexible for other data formats
# TODO: fix saved corrected file to keep the whole original dict


DATA_NAME = 'data/cais_mmlu_all_uk_TRANK'
DATASET_PATH = DATA_NAME + '.json'
CORRECTED_PATH = DATA_NAME + '_corrected.jsonl'  

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_corrected(file_path):
    if not os.path.exists(file_path):
        return []
    corrected = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                corrected.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.warn("Skipping invalid JSON line in corrected.jsonl.")
    return corrected

def append_corrected(file_path, correction):
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(correction, f, ensure_ascii=False)
        f.write('\n')

dataset = load_dataset(DATASET_PATH)
total_entries = len(dataset)

existing_corrected = load_corrected(CORRECTED_PATH)
start_index = len(existing_corrected)

def get_current_item(index):
    if index < len(dataset):
        return dataset[index]
    else:
        return None

def generate_progress_bar(current, total):
    percentage = (current / total) * 100 if total > 0 else 0
    percentage = min(percentage, 100)  
    progress_html = f"""
    <div style="width: 100%; background-color: #f3f3f3; border-radius: 5px; margin-top: 10px;">
      <div style="
          width: {percentage}%;
          height: 20px;
          background-color: #4CAF50;
          border-radius: 5px;
          text-align: center;
          line-height: 20px;
          color: white;
          transition: width 0.3s;
      ">
        {current} / {total} ({percentage:.2f}%)
      </div>
    </div>
    """
    return progress_html

def handle_vote(text_vote, text_user_text, answer_vote, answer_user_text, index, results):
    current_item = get_current_item(index)
    
    # If no more items to label
    if current_item is None:
        status_message = "✅ All items have been labeled and saved to 'corrected.jsonl'."
        progress_html = generate_progress_bar(total_entries, total_entries)
        return (
            "✅ Labeling Complete.",  # question_original
            "",                       # question_translated
            "",                       # question_corrected
            "",                       # choices_original
            "",                       # choices_translated
            "",                       # choices_corrected
            index,                    # index remains the same
            gr.update(visible=False, value=""), # hide and clear text_user_correction
            gr.update(visible=False, value=""), # hide and clear answer_user_correction
            status_message,           # output message
            progress_html             # progress bar
        )
    
    # Prepare the result entry for text
    text_result = {
        'question': current_item.get('question', ''),
        'question_translated': current_item.get('question_translated', ''),
        'question_corrected': current_item.get('question_corrected', ''),
        'question_vote': text_vote
    }
    
    if text_vote == 'None':
        if not text_user_text.strip():
            # Prompt the user to provide correction for text
            progress_html = generate_progress_bar(index, total_entries)
            return (
                current_item.get('question', ''),    # question_original
                current_item.get('question_translated', ''),  # question_translated
                current_item.get('question_corrected', ''),   # question_corrected
                current_item.get('choices', []),  # choices_original
                current_item.get('choices_translated', []),# choices_translated
                current_item.get('choices_corrected', []), # choices_corrected
                index,                                     # index remains the same
                gr.update(visible=True),                   # show text_user_correction textbox
                gr.update(visible=False),                  # hide answer_user_correction textbox
                "⚠️ Please provide the correct text.",     # output message
                progress_html                             # progress bar
            )
        text_result['question_user_corrected'] = text_user_text.strip()
    else:
        text_result['question_user_corrected'] = None
    
    answer_result = {
        'choices': current_item.get('choices', []),
        'choices_translated': current_item.get('choices_translated', []),
        'choices_corrected': current_item.get('choices_corrected', []),
        'choices_vote': answer_vote
    }
    
    if answer_vote == 'None':
        if not answer_user_text.strip():
            # Prompt the user to provide correction for answer
            progress_html = generate_progress_bar(index, total_entries)
            return (
                current_item.get('question', ''),    # question_original
                current_item.get('question_translated', ''),  # question_translated
                current_item.get('question_corrected', ''),   # question_corrected
                current_item.get('choices', []),  # choices_original
                current_item.get('choices_translated', []),# choices_translated
                current_item.get('choices_corrected', []), # choices_corrected
                index,                                     # index remains the same
                gr.update(visible=False),                  # hide text_user_correction textbox
                gr.update(visible=True),                   # show answer_user_correction textbox
                "⚠️ Please provide the correct answer.",   # output message
                progress_html                             # progress bar
            )
        
        answer_result['choices_user_corrected'] = [item.strip() for item in answer_user_text.strip().split(';') if item.strip()]
    else:
        answer_result['choices_user_corrected'] = None
    
    combined_result = {
        'index': index,
        'text': text_result,
        'answer': answer_result
    }
    
    try:
        append_corrected(CORRECTED_PATH, combined_result)
        status_message = "✅ Vote submitted successfully."
    except Exception as e:
        status_message = f"⚠️ Error saving the vote: {str(e)}"
    
    index += 1
    next_item = get_current_item(index)
    
    if next_item is None:
        status_message = "✅ All items have been labeled and saved to 'corrected.jsonl'."
        progress_html = generate_progress_bar(total_entries, total_entries)
        return (
            "✅ Labeling Complete.",  # question_original
            "",                       # question_translated
            "",                       # question_corrected
            "",                       # choices_original
            "",                       # choices_translated
            "",                       # choices_corrected
            index,                    # index remains the same
            gr.update(visible=False, value=""), # hide and clear text_user_correction
            gr.update(visible=False, value=""), # hide and clear answer_user_correction
            status_message,           # output message
            progress_html             # progress bar
        )
    
    progress_html = generate_progress_bar(index, total_entries)
    
    def join_list(lst):
        return "\n".join(lst) if isinstance(lst, list) else str(lst)
    
    return (
        next_item.get('question', ''),         # question_original
        next_item.get('question_translated', ''),       # question_translated
        next_item.get('question_corrected', ''),        # question_corrected
        join_list(next_item.get('choices', [])),  # choices_original
        join_list(next_item.get('choices_translated', [])),# choices_translated
        join_list(next_item.get('choices_corrected', [])), # choices_corrected
        index,                                      # updated index
        gr.update(visible=False, value=""),         # hide and clear text_user_correction textbox
        gr.update(visible=False, value=""),         # hide and clear answer_user_correction textbox
        status_message,                             # output message
        progress_html                               # progress bar
    )

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Text and Answer Labeling Interface")
    
    index = gr.State(start_index)
    results = gr.State(existing_corrected)
    
    with gr.Group():
        # Text Evaluation Section
        with gr.Row():
            gr.Markdown("\n ## Text Evaluation")
            question_original = gr.Textbox(label="Original Text", lines=4, interactive=False)
            question_translated = gr.Textbox(label="📜 Translated Text", lines=4, interactive=False)
            question_corrected = gr.Textbox(label="🔧 Corrected Text", lines=4, interactive=False)
            
            text_vote = gr.Radio(
                choices=["Translated", "Corrected", "None"],
                label="❓ Which translation of the text is correct?",
                value="Translated",
                interactive=True
            )
            
            text_user_correction = gr.Textbox(
                label="📝 Your Correction for Text",
                placeholder="Please enter your correction here...",
                visible=False,
                lines=3
            )
        
        with gr.Row():
            gr.Markdown("\n ## Answer Evaluation")
            choices_original = gr.Textbox(label="Original Answers", lines=4, interactive=False)
            choices_translated = gr.Textbox(label="📜 Translated Answers", lines=4, interactive=False)
            choices_corrected = gr.Textbox(label="🔧 Corrected Answers", lines=4, interactive=False)
            
            answer_vote = gr.Radio(
                choices=["Translated", "Corrected", "None"],
                label="❓ Which translation of the answer is correct?",
                value="Translated",
                interactive=True
            )
            
            answer_user_correction = gr.Textbox(
                label="📝 Your Correction for Answer",
                placeholder="Type answer choices divided by ;, like '1;2;3;4'.",
                visible=False,
                lines=3
            )
    
    with gr.Row():
        submit = gr.Button("✅ Submit Votes")
    
    output = gr.Textbox(
        label="📢 Status",
        interactive=False,
        lines=2,
        value=""
    )
    
    progress = gr.HTML(
        label="📊 Progress",
        value=""  
    )
    
    def initialize_interface(current_index, existing_results):
        if current_index < total_entries:
            current_item = get_current_item(current_index)
            progress_html = generate_progress_bar(current_index, total_entries)
            return (
                current_item.get('question', ''),      # question_original
                current_item.get('question_translated', ''),    # question_translated
                current_item.get('question_corrected', ''),     # question_corrected
                join_list(current_item.get('choices', [])),  # choices_original
                join_list(current_item.get('choices_translated', [])),# choices_translated
                join_list(current_item.get('choices_corrected', [])), # choices_corrected
                gr.update(),      # index remains the same
                gr.update(visible=False, value=""),           # hide user_correction for text
                gr.update(visible=False, value=""),           # hide user_correction for answer
                "",                # empty output
                progress_html     # initial progress bar
            )
        else:

            status_message = "✅ All items have been labeled and saved to 'corrected.jsonl'."
            progress_html = generate_progress_bar(total_entries, total_entries)
            return (
                "✅ Labeling Complete.",  # question_original
                "",                       # question_translated
                "",                       # question_corrected
                "",                       # choices_original
                "",                       # choices_translated
                "",                       # choices_corrected
                index,                    # index remains the same
                gr.update(visible=False, value=""), # hide and clear user_correction for text
                gr.update(visible=False, value=""), # hide and clear user_correction for answer
                status_message,           # output message
                progress_html             # progress bar
            )

    def join_list(lst):
        return "\n".join(lst) if isinstance(lst, list) else str(lst)

    init_question_original, init_question_translated, init_question_corrected, \
    init_choices_original, init_choices_translated, init_choices_corrected, \
    init_index, init_text_user_corr, init_answer_user_corr, \
    init_output, init_progress = initialize_interface(start_index, existing_corrected)
    
    question_original.value = init_question_original
    question_translated.value = init_question_translated
    question_corrected.value = init_question_corrected
    choices_original.value = init_choices_original
    choices_translated.value = init_choices_translated
    choices_corrected.value = init_choices_corrected
    index.value = start_index
    text_user_correction.visible = False
    answer_user_correction.visible = False
    output.value = init_output
    progress.value = init_progress
    
    def toggle_text_correction(vote_selection):
        if vote_selection == "None":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False, value="")  
    
    def toggle_answer_correction(vote_selection):
        if vote_selection == "None":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False, value="") 
    
    text_vote.change(toggle_text_correction, inputs=text_vote, outputs=text_user_correction)
    answer_vote.change(toggle_answer_correction, inputs=answer_vote, outputs=answer_user_correction)
    
    def submit_vote(text_vote_selection, text_user_text_input,
                   answer_vote_selection, answer_user_text_input,
                   current_index, voting_results):
        return handle_vote(text_vote_selection, text_user_text_input,
                          answer_vote_selection, answer_user_text_input,
                          current_index, voting_results)
    
    submit.click(
        submit_vote,
        inputs=[text_vote, text_user_correction,
                answer_vote, answer_user_correction,
                index, results],
        outputs=[question_original, question_translated, question_corrected,
                 choices_original, choices_translated, choices_corrected,
                 index, 
                 text_user_correction, answer_user_correction,
                 output, progress]
    )
    
    with gr.Row():
        reset = gr.Button("🔄 Reset Labeling")
        
        def reset_labeling():

            if os.path.exists(CORRECTED_PATH):
                os.remove(CORRECTED_PATH)
            # Reset state variables
            first_item = get_current_item(0)
            reset_progress = generate_progress_bar(0, total_entries)
            return (
                0,                                       # Reset index to 0
                [],                                      # Reset results to empty list
                first_item.get('question', ''),    # question_original
                first_item.get('question_translated', ''),  # question_translated
                first_item.get('question_corrected', ''),   # question_corrected
                join_list(first_item.get('choices', [])),  # choices_original
                join_list(first_item.get('choices_translated', [])),# choices_translated
                join_list(first_item.get('choices_corrected', [])), # choices_corrected
                gr.update(visible=False, value=""),      # hide text_user_correction
                gr.update(visible=False, value=""),      # hide answer_user_correction
                "🔄 Labeling has been reset.",           # output message
                reset_progress                           # reset progress bar
            )
        
        reset.click(
            reset_labeling,
            inputs=None,
            outputs=[index, results, question_original, question_translated, question_corrected,
                     choices_original, choices_translated, choices_corrected,
                     text_user_correction, answer_user_correction,
                     output, progress]
        )

demo.launch()