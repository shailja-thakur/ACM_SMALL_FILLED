import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeVar

import openai
import tenacity
import tiktoken

from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

from genai.schema import (
    DecodingMethod,
    HumanMessage,
    ModerationHAP,
    ModerationHAPInput,
    ModerationHAPOutput,
    ModerationParameters,
    SystemMessage,
    TextGenerationParameters,
)

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import AsyncLLMEngine
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

load_dotenv()


N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2


def read_jsonl(path: str | Path) -> list[Any]:
    """Read lines of JSON from a file (including '\n')."""
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, data: Sequence[Mapping], mode: str = "w"):
    # cannot use `dict` here as it is invariant
    with Path(path).open(mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def retry(errors: Any, max_attempts: int = 5):
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(errors),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=print,
    )


ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncClient()

    @retry(ERRORS)
    def chat_completions_with_backoff(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    def completions_with_backoff(self, *args, **kwargs):
        return self.client.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def chat_completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.completions.create(*args, **kwargs)

    async def delayed_request(
        self,
        request: dict[str, Any],
        mode: Literal["chat", "completion"],
        delay: float | None,
    ):
        """Prevent quantized rate limit:
        https://help.openai.com/en/articles/6891753-rate-limit-advice"""
        if delay is not None:
            # synchronized sleep
            time.sleep(delay)
        if mode == "chat":
            func = self.chat_completions_with_backoff_async
        else:
            func = self.completions_with_backoff_async
        return await func(**request)

    async def dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch chat completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [self.delayed_request(request, "chat", delay) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def dispatch_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        tasks = [
            self.delayed_request(request, "completion", delay) for request in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

class GenAIClient:
    def __init__(self):
        # Load environment variables for the GenAI client
        load_dotenv()
        self.client = Client(credentials=Credentials.from_env())

    def get_model_response(self, system_prompt, user_prompt, model_id="meta-llama/llama-3-8b-instruct", max_new_tokens=400, min_new_tokens=30, temperature=0.7, top_k=50, top_p=1):
        parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE, 
            max_new_tokens=max_new_tokens, 
            min_new_tokens=min_new_tokens, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user_prompt),
        ]
        
        response = self.client.text.chat.create(
            model_id=model_id,
            messages=messages,
            parameters=parameters,
        )

        return response.results[0].generated_text
    
class VLLMClient_batch:
    def __init__(self, model_name="microsoft/phi-4"):
        #logger.info(f"Initializing VLLMClient with model_name={model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_args = {
            "model": model_name,
            "gpu_memory_utilization": 0.85,
        }
        #logger.info(f"Creating LLM with args: {llm_args}")
        self.llm = LLM(**llm_args)
        #logger.info("LLM initialized successfully")
        self.default_sampling_params = SamplingParams(max_tokens=8192)
        #logger.debug("Default sampling params set: max_tokens=8192")

    def get_model_response(self, system_prompt, user_prompt, model_id="microsoft/phi-4", max_new_tokens=400, min_new_tokens=30, temperature=0.2, top_k=50, top_p=0.9, repetition_penalty=1.05):
        #logger.info(f"get_model_response called with max_new_tokens={max_new_tokens}, temperature={temperature}")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty
        )
        #logger.debug(f"Sampling params: {sampling_params}")

        # Single prompt case
        if isinstance(system_prompt, str) and isinstance(user_prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            #logger.debug(f"Single prompt (chat format): {text}")
            outputs = self.llm.generate([text], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
            #logger.debug(f"Single response: {response[:200]}...")
            return response

        # Batch prompt case
        elif isinstance(system_prompt, list) and isinstance(user_prompt, list):
            if len(system_prompt) != len(user_prompt):
                raise ValueError("system_prompt and user_prompt lists must have the same length")
            combined_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for sys, usr in zip(system_prompt, user_prompt)
            ]
            #logger.info(f"Batch processing {len(combined_prompts)} prompts")
            #for i, prompt in enumerate(combined_prompts[:3]):
            #    logger.debug(f"Batch prompt {i} (full, chat format): {prompt}")
            #if len(combined_prompts) > 3:
            #    logger.debug(f"Truncated {len(combined_prompts) - 3} additional prompts")
            outputs = self.llm.generate(combined_prompts, sampling_params=sampling_params)
            results = [output.outputs[0].text for output in outputs]
            #for i, response in enumerate(results[:5]):
            #    logger.debug(f"Batch response {i} (full): {response}")
            #if len(results) > 5:
            #    logger.debug(f"Truncated {len(results) - 5} additional responses")
            #logger.info(f"Batch completed with {len(results)} responses")
            return results

        else:
            raise ValueError("system_prompt and user_prompt must both be strings or both be lists")



class VLLMClient_mistral:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", max_model_len=4096):
        logger.info(f"Initializing VLLMClient with model_name={model_name}, max_model_len={max_model_len}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Explicitly set 8 GPUs
        llm_args = {
            "model": model_name,
            "gpu_memory_utilization": 0.7,
            "max_model_len": max_model_len,
            "tensor_parallel_size": 8,
            "disable_custom_all_reduce": True,  # Fallback to NCCL
        }
        logger.info(f"Creating LLM with args: {llm_args}")
        try:
            self.llm = LLM(**llm_args)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
        self.default_sampling_params = SamplingParams(max_tokens=512)

    def get_model_response(self, system_prompt, user_prompt, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                           max_new_tokens=512, min_new_tokens=30, temperature=0.2, top_k=50, top_p=0.9,
                           repetition_penalty=1.05):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        res = self.llm.chat(messages=messages, sampling_params=sampling_params)
        return res[0].outputs[0].text




class VLLMClient:
    def __init__(self, model_name="mistralai/mixtral-8x7b-instruct-v01"):
        # Initialize the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name)
        
    def get_model_response(self, system_prompt, user_prompt, model_id="mistralai/mixtral-8x7b-instruct-v01", max_new_tokens=400, min_new_tokens=30, temperature=0.7, top_k=50, top_p=0.8, repetition_penalty=1.05):
        # Setup the sampling parameters for generation
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            #repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens  # Adjusting the total generation length
        )
        
        # Prepare the messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Tokenize the input using the chat template format
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate output using the vLLM model
        outputs = self.llm.generate([text], sampling_params)
        
        # Extract and return the generated text from the output
        generated_text = outputs[0].outputs[0].text
        print(generated_text)
        return generated_text

    

# class HuggingFaceClient:
#     def __init__(self):
#         # Load environment variables for the GenAI client
        
#     def get_model_response(self, system_prompt, user_prompt, model_id="mistralai/mixtral-8x7b-instruct-v01", max_new_tokens=400, min_new_tokens=30, temperature=0.7, top_k=50, top_p=1):
#         parameters = TextGenerationParameters(
#             decoding_method=DecodingMethod.SAMPLE, 
#             max_new_tokens=max_new_tokens, 
#             min_new_tokens=min_new_tokens, 
#             temperature=temperature, 
#             top_k=top_k, 
#             top_p=top_p
#         )

#         messages = [
#             SystemMessage(content="You are a helpful assistant."),
#             HumanMessage(content=user_prompt),
#         ]
        
#         response = self.client.text.chat.create(
#             model_id=model_id,
#             messages=messages,
#             parameters=parameters,
#         )

#         return response.results[0].generated_text


# class GenAIClient:
#     def __init__(self):
#         # Load environment variables for the GenAI client
#         load_dotenv()
#         self.client = Client(credentials=Credentials.from_env())
    
    
#     @retry(ERRORS)
#     def text_generation_with_backoff(self, *args, **kwargs):
#         return self.client.text.chat.create(*args, **kwargs)

#     @retry(ERRORS)
#     async def text_generation_with_backoff_async(self, *args, **kwargs):
#         return await asyncio.to_thread(self.client.text.chat.create, *args, **kwargs)

#     async def delayed_request(
#         self,
#         request: dict[str, Any],
#         mode: Literal["text_generation"],
#         delay: float | None,
#     ):
#         """Prevent rate limit issues with delay handling."""
#         if delay is not None:
#             time.sleep(delay)

#         if mode == "text_generation":
#             func = self.text_generation_with_backoff_async

#         return await func(**request)

#     async def dispatch_text_generation(
#         self,
#         requests: list[dict[str, Any]],
#         delay: float | None = None,
#     ):
#         """Dispatch text generation requests asynchronously.
#         Args:
#             requests: a list of API argument names to values.
#             delay: interval between requests.
#         """

#         tasks = [self.delayed_request(request, "text_generation", delay) for request in requests]
#         return await asyncio.gather(*tasks, return_exceptions=True)



# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    # encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_fingerprint(*args: Any, hash_length: int | None = None) -> str:
    combined = "".join(map(str, args))
    content = hashlib.sha256(combined.encode()).hexdigest()
    if hash_length is not None:
        content = content[:hash_length]
    return content


def find_code_blocks(response: str, tag: str | None = None) -> list[str]:
    """Find all enclosed code blocks in the response, optionally filtering by language tag."""
    all_indices = find_codeblock_indices(response, tag)
    return [response[start:end].strip() for start, end in all_indices]


def find_codeblock_indices(
    response: str, tag: str | None = None
) -> list[tuple[int, int]]:
    """Find all enclosed code blocks in the response, optionally filtering by language tag."""
    all_indices: list[tuple[int, int]] = []
    search_start = (
        0  # Variable to keep track of where to start searching for the next code block
    )

    while "```" in response[search_start:]:
        # Find the start of the code block (excluding the backticks)
        code_start_index = response.find("```", search_start) + 3

        # Find the end of the language tag line (or the start of the code if no tag line)
        code_start_endline = response.find("\n", code_start_index)
        if code_start_endline == -1:  # Handle case where there's no newline after ```
            code_start_endline = code_start_index

        # Extract the language tag (if any)
        extracted_tag = response[code_start_index:code_start_endline].strip()

        # Adjust the start index if a language tag is found
        if extracted_tag:
            actual_code_start = code_start_endline + 1
        else:
            actual_code_start = code_start_index

        # Find the end of the code block
        code_end_index = response.find("```", actual_code_start)
        if code_end_index == -1:
            break  # Exit if there's no closing ```

        # Extract the code
        # code = response[actual_code_start:code_end_index].strip()

        # Check if the extracted code block matches the requested language tag (if any)
        if tag is None or extracted_tag.lower() == tag.lower():
            all_indices.append((actual_code_start, code_end_index))

        # Update the search_start to look for the next code block
        search_start = code_end_index + 3

    return all_indices


def remove_comments_from_code_blocks(
    content: str,
) -> str:
    code_blocks = find_codeblock_indices(content)
    # Current index in the original content for tracking purposes
    current_index = 0
    # Buffer to store the new content
    new_content: list[str] = []
    # Iterate over each code block
    for start, end in code_blocks:
        # Append the content before this code block
        new_content.append(content[current_index:start])

        # Extract the code block content
        code_block_content = content[start:end]

        # Split into lines, process, and rejoin
        lines = code_block_content.splitlines(keepends=True)
        kept_lines = list[str]()

        i = 0
        while i < len(lines):
            if (
                i != 0
                and i + 1 < len(lines)
                and lines[i].strip() == ""
                and lines[i + 1].lstrip().startswith("#")
            ):
                i += 2
                continue
            if lines[i].lstrip().startswith("#"):
                i += 1
                continue
            kept_lines.append(lines[i])
            i += 1

        # Join the processed lines and add to the modified blocks list
        modified_block_content = "".join(kept_lines)
        new_content.append(modified_block_content)

        # Update current index
        current_index = end

    # Add the remaining part of the original content after the last code block
    new_content.append(content[current_index:])

    # Join all parts to form the final modified content
    return "".join(new_content)


def infer_prompt_template(tokenizer_name: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    template = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "{instruction}"},
            {"role": "assistant", "content": "{response}"},
        ],
        tokenize=False,
    )
    end_index = template.rindex("{response}") + len("{response}")
    template = template[:end_index]
    return template
