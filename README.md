<h1 align="center">When LLMs Meet Cunning Texts: A Fallacy Understanding Benchmark for Large Language Models</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2402.11100">Paper</a> | <a href="https://thukelab.github.io/FLUB/">Webpage</a>
</p>

### Requirements

- transformers ~= 4.35.0
- vllm ~= 0.2.2
- openai == 0.28.0
- scikit-learn
- pandas
- numpy
- tqdm

### How to run

#### Step 1: Model Inference

For local models (Qwen-72B-Chat, Yi-34B-Chat, Baichuan2-13B-Chat, etc.), here is an example:

```shell
python tasks.py \
    --model_name Qwen-72B-Chat \
    --tp_size 8 \
    --gpu_memory_utilization 0.9 \
    --fewshot 0
```

For API models (GPT-4-Turbo, ERNIE-Bot-4.0, etc.), here is another example:

```shell
python tasks.py \
    --model_name gpt-4-1106-preview \
    --is_api \
    --num_processes 32 \
    --fewshot 0
```

#### Step 2: Automatic Evaluation

Please run

```shell
python evaluation.py
```

#### Step 3: Compute Metrics

Please run

```shell
python analysis.py
```

All metrics would be saved to `metrics.tsv`.

### Metadata and Data Format

The Croissant metadata of FLUB is at [FLUB_croissant_metadata](https://github.com/THUKElab/FLUB/blob/main/FLUB_croissant_metadata.json)

The data format of FLUB is as follows:

```json
{
  "text": "The input cunning text",
  "is_question": "Is the input cunning text a question?",
  "type": "The cunning type of the input text for the Cunning Type Classification task.",
  "explanation": "The correct explanation of the input text for the Fallacy Explanation task.",
  "id": "The id of each data sample",
  "options": {
    "A": "The candidate answer 1 for the input text (question)",
    "B": "The candidate answer 2 for the input text (question)",
    "C": "The candidate answer 3 for the input text (question)",
    "D": "The candidate answer 4 for the input text (question)"
  },
  "answer": "The correct answer for the Answer Selection (Multiple Choice) task."
}
```



### Citation

Please consider citing this paper if you use the `code` or `data` from our work. Thanks a lot :)

```
@article{li2024llms,
  title={When llms meet cunning texts: A fallacy understanding benchmark for large language models},
  author={Li, Yinghui and Zhou, Qingyu and Luo, Yuanzhen and Ma, Shirong and Li, Yangning and Zheng, Hai-Tao and Hu, Xuming and Yu, Philip S},
  journal={arXiv preprint arXiv:2402.11100},
  year={2024}
}
```

### License
The  `data` and our `code` are both subject to the license of [Creative Commons Attribution 4.0 International (CC BY 4.0)](./LICENSE).
