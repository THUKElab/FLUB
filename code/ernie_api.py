import requests
import json
import functools

api_key = "xxx"
secret_key = "xxx"

base_urls = {
    "ernie-bot-40": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=",
    "ernie-bot-35": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=",
    "ernie-bot-35-turbo-0922": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token="
}

@functools.lru_cache()
def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def call_ernie_api(prompt, model_name, **kwargs):  
    base_url = base_urls[model_name.lower()]
    url = base_url + get_access_token()
    max_retry = kwargs.pop("max_retry", 3)
    retry_count = 0
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
        ],
        **kwargs
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            output = json.loads(response.text)
            return output["result"]
        except Exception as e:
            print(f"Exception: {e} / Prompt: {prompt} / Model: {model_name} / Retry: {retry_count}")
            if retry_count >= max_retry:
                raise e
            retry_count += 1
            continue
    

if __name__ == '__main__':
    print(call_ernie_api(prompt="你是谁？"))