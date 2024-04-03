# coding=utf-8
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import copy
import random
random.seed(0)
from vllm_wrapper import VLLMWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--is_api", action="store_true")
parser.add_argument("--tp_size", type=int, default=1)
parser.add_argument("--swap_space", type=int, default=4)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
parser.add_argument("--num_processes", type=int, default=16)
parser.add_argument("--fewshot", type=int, default=0)
args, _ = parser.parse_known_args()

model_name = args.model_name
fewshot = args.fewshot
model = None

if not args.is_api:
    model_base_dir = "xxx"
    model_path = os.path.join(model_base_dir, args.model_name)
    if not os.path.exists(os.path.join(model_path, "config.json")):
        model_path = os.path.join(model_path, "main")
    model = VLLMWrapper(model=model_path, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=args.tp_size, swap_space=args.swap_space, model_name=args.model_name)

data_file = "ruozhiba.json"
output_dir = "outputs"
if args.fewshot:
    output_dir = "outputs_fewshot"
prompt_dir = "prompts"
os.makedirs(output_dir, exist_ok=True)
with open(data_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
candidates = "，".join(sorted(set([item["type"] for item in data if isinstance(item["type"], str)])))

tasks = ["classification", "selection", "explanation"]
# tasks = ["classification_nocot", "selection_nocot"]

decoding_params = dict(temperature=0.7, top_p=0.8, max_tokens=1024)
num_processes = args.num_processes
if "ernie" in args.model_name.lower():
    decoding_params = dict(temperature=0.7, top_p=0.8)
    from ernie_api import call_ernie_api
else:
    from openai_api import call_openai_api

shot_templates = {
    "classification": "输入：{sentence}\n分类：{answer}",
    "selection": "输入：{sentence}\n选项：\n{options}\n答案：{answer}",
    "explanation_q": "输入问题：{sentence}\n回答：{answer}",
    "explanation_nq": "输入句子：{sentence}\n解释：{answer}"
}

fewshot_data = {key: {} for key in shot_templates}
for item in data:
    _id = item["id"]
    _type = item["type"]
    sentence = item["text"]
    options = "\n".join([f"{option}: {content}" for option, content in item["options"].items()])
    answer = item["answer"]
    explanation = item["explanation"]
    fewshot_data["selection"][_id] = dict(sentence=sentence, options=options, answer=answer)
    if isinstance(_type, str):
        fewshot_data["classification"][_id] = dict(sentence=sentence, answer=_type)
    if item["is_question"]:
        fewshot_data["explanation_q"][_id] = dict(sentence=sentence, answer=explanation)
    else:
        fewshot_data["explanation_nq"][_id] = dict(sentence=sentence, answer=explanation)

for task in tasks:
    task_name = task
    if args.fewshot:
        task = task_name + "_fs"
    print(f"********** Task: {task} | Model: {model_name} **********")
    prompt_files = {
        "": f"{task}.txt",
        "q": f"{task}_q.txt",
        "nq": f"{task}_nq.txt"
    }
    prompt_templates = {}
    for key, file in prompt_files.items():
        file = os.path.join(prompt_dir, file)
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                prompt_templates[key] = f.read()
    ids = set()
    if args.fewshot:
        task = f"{task_name}_{args.fewshot}shot"
    output_file = os.path.join(output_dir, f"{task}_output_{model_name.lower()}.jsonl")
    task_data = copy.deepcopy(data)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        ids = set([item["id"] for item in items])
        # task_data = [item for item in data if item["id"] not in ids]
    prompts = []
    outputs = []
    for i, item in tqdm(enumerate(task_data), total=len(task_data), desc="Prompt"):
        _id = item["id"]
        # print(prompt_templates)
        fs_match_name = task_name
        if task_name == "explanation":
            fs_match_name = fs_match_name + ("_q" if item["is_question"] else "_nq")
        shot_template = shot_templates.get(fs_match_name)
        if item["is_question"]:
            prompt_template = prompt_templates.get("q", prompt_templates.get("", None))
        else:
            prompt_template = prompt_templates.get("nq", prompt_templates.get("", None))
        slots = dict(
            sentence=item["text"]
        )
        answer = None
        if task.startswith("classification"):
            slots["candidates"] = candidates
            _type = item["type"]
            if not isinstance(_type, str):
                continue
            answer = _type
        elif task.startswith("selection"):
            options = [f"{option}: {content}" for option, content in item["options"].items()]
            slots["options"] = "\n".join(options)
            answer = item["answer"]
        else:
            answer = item["explanation"]
        if args.fewshot:
            step_fewshot_data = copy.deepcopy(fewshot_data[fs_match_name])
            step_fewshot_data.pop(_id)
            sorted_data = list(map(lambda x: x[1], sorted(step_fewshot_data.items(), key=lambda x: x[0])))
            selected_data = random.sample(sorted_data, k=args.fewshot)
            slots["shots"] = "\n\n".join([shot_template.format(**fs_item) for fs_item in selected_data])
        prompt = prompt_template.format(**slots)
        prompts.append(prompt)
        output = {
            "id": _id,
            "text": item["text"],
            "prompt": prompt,
            "response": None,
            "answer": answer
        }
        outputs.append(output)
    
    gpt4_output_file = os.path.join(output_dir, f"{task}_output_gpt-4-1106-preview.jsonl")
    if args.fewshot and os.path.exists(gpt4_output_file):
        print("[Warning] Reuse GPT-4 Output File!!!")
        with open(gpt4_output_file, "r", encoding="utf-8") as f:
            outputs = [json.loads(line) for line in f.readlines()]
        prompts = [output["prompt"] for output in outputs]
        for output in outputs:
            output["response"] = None
    
    prompts = [prompt for i, prompt in enumerate(prompts) if outputs[i]["id"] not in ids]
    outputs = [output for output in outputs if output["id"] not in ids]
    print(f"Origin {len(task_data)} Samples | Remain {len(outputs)} Samples")
    if len(outputs) <= 0:
        continue
    
    fw = open(output_file, "a+", encoding="utf-8")
    if args.is_api:
        def func(prompt):
            if "ernie" in model_name:
                return call_ernie_api(prompt, model_name, **decoding_params)
            else:
                return call_openai_api(prompt, model_name, **decoding_params)
        with Pool(processes=num_processes) as pool:
            for i, response in tqdm(enumerate(pool.imap(func, prompts)), total=len(prompts), desc="Model Inference"):
                output = outputs[i]
                output["response"] = response
                fw.write(json.dumps(output, ensure_ascii=False) + "\n")
    else:
        prompts = [item["prompt"] for item in outputs]
        responses, _ = model.chat(prompts, history=None, **decoding_params)
        assert len(responses) == len(outputs)
        for i, response in enumerate(responses):
            output = outputs[i]
            output["response"] = response
            fw.write(json.dumps(output, ensure_ascii=False) + "\n")
    fw.close()

        
