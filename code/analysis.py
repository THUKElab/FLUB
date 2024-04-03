import os
import re
import json
import pandas as pd

data_file = "ruozhiba.json"
with open(data_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
candidates = sorted(set([item["type"] for item in data if isinstance(item["type"], str)]))
print(f"{candidates=}")

# base_dir = "outputs"
# evaluation_dir = "evaluations"
base_dir = "outputs_fewshot"
evaluation_dir = "evaluations_fewshot"
metrics = {}

options = "ABCD"

def verbalize(text, task):
    if task.startswith("classification"):
        match = re.search(f"分类：({'|'.join(candidates)})", text)
        if match is not None:
            return match.group(1)
        match2 = re.search("|".join(candidates), text)
        if match2 is not None:
            return match2.group(0)
        return ""
    elif task.startswith("selection"):
        match = re.search(r"答案：(.+)", text)
        if match is not None:
            match_str = match.group(1)
            if match_str[0] in options:
                return match_str[0]
        match2 = re.search(f"[{options}]", text)
        if match2 is not None:
            return match2.group()
        return ""
    else:
        raise NotImplementedError

def get_score(prediction, answer, task):
    return float(prediction == answer)

def match_score(evaluation):
    match = re.search(r"\[\[(\d+)\]\]", evaluation)
    if match is None:
        return None
    score = int(match.group(1))
    assert 1 <= score <= 10, f"{evaluation}"
    return score

for file in os.listdir(base_dir):
    if not file.endswith(".jsonl"):
        continue
    match = re.match(r"(.+)_output_(.+).jsonl", file)
    task, model_name = match.groups()
    print(f"********** Analysis: {file} **********")
    metrics.setdefault(model_name, {})
    score = 0
    if task.startswith("explanation"):
        valid_count = 0
        path = os.path.join(evaluation_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        for item in items:
            response = item["evaluation_response"]
            item_score = match_score(response)
            if item_score is not None:
                score += item_score
                valid_count += 1
        score = score / valid_count
        print(f"{score=}, {len(items)=}, {valid_count=}")
        
    else:
        score = 0
        path = os.path.join(base_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        for item in items:
            response, answer = item["response"], item["answer"]
            prediction = verbalize(response, task)
            score += get_score(prediction, answer, task)
        score = score / len(items)
    metrics[model_name][task] = score

model_name_mapping = {
    "baichuan2-7b-chat": "Baichuan2-7B-Chat",
    "baichuan2-13b-chat": "Baichuan2-13B-Chat",
    "chatglm3-6b": "ChatGLM3-6B",
    "ernie-bot-35-turbo-0922": "ERNIE-Bot-3.5-Turbo",
    "ernie-bot-35": "ERNIE-Bot-3.5",
    "ernie-bot-40": "ERNIE-Bot-4.0",
    "gpt-4-1106-preview": "GPT-4-Turbo",
    "gpt-35-turbo-1106": "GPT-3.5-Turbo",
    "qwen-7b-chat": "Qwen-7B-Chat",
    "qwen-14b-chat": "Qwen-14B-Chat",
    "qwen-72b-chat": "Qwen-72B-Chat",
    "yi-6b-chat": "Yi-6B-Chat",
    "yi-34b-chat": "Yi-34B-Chat",
}
# columns = ["model_name", "classification", "selection", "explanation", "classification_nocot", "selection_nocot"]
columns = ["model_name"] + [f"{task}_{shot}shot" for task in ["classification", "selection", "explanation"] for shot in [1, 2, 5]]
output = pd.DataFrame(columns=columns)
for model_name, task_metrics in metrics.items():
    output.loc[len(output)] = {"model_name": model_name_mapping[model_name.lower()], **task_metrics}
# output = output.sort_values(by="explanation", ascending=False)
for column in output.columns:
    if column.startswith("classification") or column.startswith("selection"):
        output[column] = output[column].map(lambda x: f"{round(x * 100, 2):.2f}")
    elif column.startswith("explanation"):
        output[column] = output[column].map(lambda x: f"{round(x, 3):.3f}")

# output.to_csv("metrics_v2.tsv", sep="\t", index=False)
output.to_csv("metrics_v3.tsv", sep="\t", index=False)