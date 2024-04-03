import os
from typing import List, Optional, Union, Tuple, Dict

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter


class VLLMWrapper:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        self.print_interval = kwargs.pop("print_interval", 100)
        self.model_name = kwargs.pop("model_name", "default").lower()
        print(f"{self.model_name=}")
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer
    
    @property
    def tokenizer(self):
        return self.llm_engine.tokenizer
    
    @property
    def config(self):
        return self.llm_engine.model_config.hf_config

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        if prompts is not None:
            num_requests = len(prompts)
        else:
            num_requests = len(prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]
            self._add_request(prompt, sampling_params, token_ids)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        # if use_tqdm:
        #     num_requests = self.llm_engine.get_num_unfinished_requests()
        #     pbar = tqdm(total=num_requests, desc="Processed prompts")
        num_requests = self.llm_engine.get_num_unfinished_requests()
        finished_requests = 0
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        # pbar.update(1)
                        finished_requests += 1
                        if finished_requests % self.print_interval == 0:
                            print(f"Processed Prompts: {finished_requests} / {num_requests}")
        # if use_tqdm:
        #     pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
    
    def build_context_chatml(
        self,
        query: List[str],
        history: Optional[List[List[str]]] = None,
        system_prompt: Optional[str] = None
    ):
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        if history is None or len(history) <= 0:
            history = [[
                    {"role": "system", "content": system_prompt},
                ] if system_prompt is not None else []
                for i in range(len(query))
            ]
        for single_history, text in zip(history, query):
            single_history.append({"role": "user", "content": text})
        prompts = []
        for single_history in history:
            prompt_list = []
            for item in single_history:
                role, content = item["role"], item["content"]
                prompt_list.append(f"{im_start}{role}\n{content}{im_end}")
            prompt_list.append(f"{im_start}assistant\n")
            prompts.append("\n".join(prompt_list))
        return prompts, history
    
    def build_context_baichuan(
        self,
        query: List[str],
        history: Optional[List[List[str]]] = None,
        system_prompt: Optional[str] = None
    ):
        user_token, assistant_token = self.tokenizer.convert_ids_to_tokens([195, 196])
        if system_prompt is None:
            system_prompt = ""
        if history is None or len(history) <= 0:
            history = [[
                    {"role": "system", "content": system_prompt},
                ]
                for i in range(len(query))
            ]
        for single_history, text in zip(history, query):
            single_history.append({"role": "user", "content": text})
        # prompts = []
        # for single_history in history:
        #     prompt_list = []
        #     for item in single_history:
        #         role, content = item["role"], item["content"]
        #         if role == "system":
        #             system_prompt = content
        #             continue
        #         special_token = user_token if role == "user" else assistant_token
        #         prompt_list.append(f"{special_token}{content}")
        #     prompt_list.append(assistant_token)
        #     prompts.append(system_prompt + "".join(prompt_list))
        # return prompts, history
        user_token_id, assistant_token_id = 195, 196
        prompt_token_ids = []
        for single_history in history:
            token_ids = []
            for item in single_history:
                role, content = item["role"], item["content"]
                if role == "system":
                    system_prompt = content
                    continue
                special_token_id = user_token_id if role == "user" else assistant_token_id
                token_ids.extend([special_token_id] + self.tokenizer.encode(content))
            token_ids.append(assistant_token_id)
            system_token_ids = self.tokenizer.encode(system_prompt)
            prompt_token_ids.append(system_token_ids + token_ids)
        return prompt_token_ids, history
    
    def build_context_chatglm(
        self,
        query: List[str],
        history: Optional[List[List[str]]] = None,
        system_prompt: Optional[str] = None
    ):
        if history is None or len(history) <= 0:
            history = [
                ([{"role": "system", "content": system_prompt}] 
                if system_prompt is not None and len(system_prompt) > 0 else [])
                for i in range(len(query))
            ]
        for single_history, text in zip(history, query):
            single_history.append({"role": "user", "content": text})
        prompt_token_ids = []
        for single_history in history:
            token_ids = []
            for item in single_history:
                role, content = item["role"], item["content"]
                token_ids.append(self.tokenizer.get_command(f"<|{role}|>"))
                token_ids.extend(self.tokenizer.encode("\n"))
                token_ids.extend(self.tokenizer.encode(content))
            token_ids.append(self.tokenizer.get_command(f"<|assistant|>"))
            token_ids.extend(self.tokenizer.encode("\n"))
            prompt_token_ids.append(token_ids)
        return prompt_token_ids, history
    
    def build_history(
        self,
        query: List[str],
        history: Optional[List[List[str]]] = None,
        system_prompt: Optional[str] = None
    ):
        if history is None or len(history) <= 0:
            history = [
                ([{"role": "system", "content": system_prompt}] 
                if system_prompt is not None and len(system_prompt) > 0 else [])
                for i in range(len(query))
            ]
        for single_history, text in zip(history, query):
            single_history.append({"role": "user", "content": text})
        return history
    
    def build_context_llama(
        self,
        history: List[List[str]]
    ):
        # Reference: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        prompt_token_ids = []
        for single_history in history:
            if single_history[0]["role"] == "system":
                single_history = [
                    {"role": single_history[1]["role"], "content": B_SYS + single_history[0]["content"] + E_SYS + single_history[1]["content"]}
                ] + single_history[2:]
            token_ids = []
            for prompt, answer in zip(single_history[::2], single_history[1::2]):
                encoding_ids = self.tokenizer.encode(f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ", add_special_tokens=True)
                token_ids.extend(encoding_ids)
                token_ids.append(self.tokenizer.eos_token_id)
            encoding_ids = self.tokenizer.encode(f"{B_INST} {single_history[-1]['content'].strip()} {E_INST}", add_special_tokens=True)
            token_ids.extend(encoding_ids)
            prompt_token_ids.append(token_ids)
        return prompt_token_ids
    
    def chat(
        self,
        query: Optional[Union[str, List[str]]],
        history: Optional[List[List[Dict[str, str]]]] = None,
        stop: Optional[List[str]] = None,
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[str], List[List[Dict[str, str]]]]:
        if stop is None:
            stop = []
        if isinstance(query, str):
            query = [query]
        use_tqdm = kwargs.pop("use_tqdm", True)
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        else:
            for key, value in kwargs.items():
                setattr(sampling_params, key, value)
        prompts, prompt_token_ids = None, None
        if self.config.model_type == "llama" and "llama" in self.model_name:
            stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop)
            stop_token_ids += [self.tokenizer.eos_token_id]
            sampling_params.stop_token_ids = stop_token_ids
            history = self.build_history(query, history, system_prompt)
            prompt_token_ids = self.build_context_llama(history)
        elif self.config.model_type in ["qwen", "llama"]:
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            stop += [im_start, im_end]
            if system_prompt is None:
                if self.config.model_type == "qwen":
                    system_prompt = "You are a helpful assistant."
                    sampling_params.stop = stop
                else:
                    system_prompt = "You are a helpful assistant"
                    sampling_params.stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop)
            prompts, history = self.build_context_chatml(query, history, system_prompt)
        elif self.config.model_type == "baichuan":
            # user_token, assistant_token = self.tokenizer.convert_ids_to_tokens([195, 196])
            # prompts, history = self.build_context_baichuan(query, history, system_prompt)
            # stop += [user_token, assistant_token]
            # sampling_params.stop = stop
            user_token_id, assistant_token_id = 195, 196
            prompt_token_ids, history = self.build_context_baichuan(query, history, system_prompt)
            stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop)
            stop_token_ids += [user_token_id, assistant_token_id]
            sampling_params.stop_token_ids = stop_token_ids
        elif self.config.model_type == "chatglm":
            stop_token_ids = [self.tokenizer.get_command(word) for word in stop]
            stop_token_ids += [self.tokenizer.get_command(f"<|{role}|>") for role in ["system", "user", "assistant"]] + [self.tokenizer.eos_token_id]
            sampling_params.stop_token_ids = stop_token_ids
            prompt_token_ids, history = self.build_context_chatglm(query, history, system_prompt)
        elif self.config.model_type == "mistral":
            if getattr(self.tokenizer, "chat_template", None) is None:
                self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop)
            stop_token_ids += [self.tokenizer.eos_token_id]
            sampling_params.stop_token_ids = stop_token_ids
            history = self.build_history(query, history, system_prompt)
            prompt_token_ids = []
            for single_history in history:
                token_ids = self.tokenizer.apply_chat_template(single_history)
                prompt_token_ids.append(token_ids)
        else:
            raise NotImplementedError
        outputs = self.generate(prompts=prompts, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=use_tqdm)
        response = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            response.append(text)
            history[i].append({"role": "assistant", "content": text})
        return response, history
    
    def base_generate(
        self,
        prompts: Optional[Union[str, List[str]]],
        stop: Optional[List[str]] = None,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> List[str]:
        if stop is None:
            stop = []
        if isinstance(prompts, str):
            prompts = [prompts]
        use_tqdm = kwargs.pop("use_tqdm", True)
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        else:
            for key, value in kwargs.items():
                setattr(sampling_params, key, value)
        # if self.config.model_type == "qwen":
        #     stop += ["<|endoftext|>"]
        sampling_params.stop = stop
        outputs = self.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
        response = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            response.append(text)
        return response
