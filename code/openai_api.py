import openai
import requests

openai.api_base = "xxx"
openai.api_key = "xxx"
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

system_prompt = "You are a helpful assistant."

def call_openai_api(prompt, model_name, **kwargs):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    max_retry = kwargs.pop("max_retry", 3)
    retry_count = 0
    while True:
        try:
            completion = openai.ChatCompletion.create(
                engine=model_name,
                model=model_name,
                messages=messages,
                **kwargs
            )
            response = completion["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            print(f"Exception: {e} / Prompt: {prompt} / Model: {model_name} / Retry: {retry_count}")
            if retry_count >= max_retry:
                return None
            retry_count += 1
            continue

