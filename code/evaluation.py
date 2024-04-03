import os
import json
from glob import glob
from openai_api import call_openai_api
from tqdm import tqdm
from multiprocessing import Pool

prompt_template = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

[Question]
{question}

[The Start of Reference Answer]
{answer}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]
""".strip()

decoding_params = dict(temperature=0.3, top_p=0.7, max_tokens=1024)
num_processes = 16

question_template = "以下的句子或问题存在不合理或幽默之处，请简要地解释其中不合理或幽默之处：\n{text}"
model_name = "gpt-4-1106-preview"

# evaluation_dir = "evaluations"
# input_dir = "outputs"
evaluation_dir = "evaluations_fewshot"
input_dir = "outputs_fewshot"

def func(line):
    item = json.loads(line)
    text, answer, response = item["text"], item["answer"], item["response"]
    question = question_template.format(text=text)
    prompt = prompt_template.format(question=question, answer=answer, response=response)
    item["evaluation_prompt"] = prompt
    response = call_openai_api(prompt, model_name, **decoding_params)
    item["evaluation_response"] = response
    return item

os.makedirs(evaluation_dir, exist_ok=True)
input_files = glob(f"{input_dir}/explanation_*output*")
for file in input_files:
    name = os.path.basename(file)
    print(f"********** Evaluation: {name} **********")
    output_file = os.path.join(evaluation_dir, name)
    processed_lines = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            processed_lines = len(lines)
    
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[processed_lines:]
    if len(lines) <= 0:
        print("Done!")
        continue
    fw = open(output_file, "a+", encoding="utf-8")
    with Pool(num_processes) as pool:
        for item in tqdm(pool.imap(func, lines), total=len(lines), desc="GPT-4 Evaluation"):
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    fw.close()
        