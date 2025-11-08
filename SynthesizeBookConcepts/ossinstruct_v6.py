import os
import json
import textwrap
from pathlib import Path
import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
import sys
from tqdm.auto import tqdm
from transformers import HfArgumentParser
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

#sys.path.append('/proj/data-eng/users/shailja/SynthesizeBookProgConcepts/selfcodealign/src/')
#import star_align
from utils import compute_fingerprint,GenAIClient, OpenAIClient, VLLMClient, HuggingFaceClient

InstructMode = Literal["I->R", "S->C", "C->I"]

LANGUAGE_MAP = {
    "cpp": "C++",
    "java": "Java",
    "php": "PHP",
    "python": "Python",
    "rust": "Rust",
    "typescript": "TypeScript",
}

@dataclass(frozen=True)
class Args:
    seed_data_file: str = field(
        metadata={"help": "Path to the text_chunks.json file"}
    )
    
    model: str
    instruct_mode: InstructMode
    model_type: str

    seed_code_start_index: int = field(default=0)
    max_new_data: int =field(default=10)
    continue_from: str | None = field(default=None)

    seed: int = field(default=3407)

    temperature: float = field(default=0.7)
    max_output_tokens: int = field(default=1536)
    prompting_mode: Literal["chat", "completion"] = field(default="completion")

    num_batched_requests: int = field(
        default=1, metadata={"help": "Number of requests to send concurrently"}
    )
    num_seed_examples: int = field(
        default=10, metadata={"help": "Number of requests to send concurrently"}
    )
    num_sample_per_request: int = field(
        default=32, metadata={"help": "Number of samples to generate per request"}
    )
    sleep: float | None = field(
        default=None, metadata={"help": "Sleep between requests in seconds"}
    )
    delay: float | None = field(
        default=None, metadata={"help": "Delay between batched requests in seconds"}
    )

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )
    save_dir: str = field(default="./")

    def fingerprint(self) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = (
            self.seed_data_file,
            self.seed,
            self.prompting_mode,
            self.temperature,
            self.model,
            self.max_output_tokens,
        )
        return compute_fingerprint(*args, hash_length=5)


import textwrap  # Added for chunking long examples


def extract_json_block(response):
    # Match the pseudo-JSON block within triple backticks or similar markers
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, response, re.DOTALL)
    
    if match:
        return match.group(1)  # Return the matched JSON-like block
    return None

def parse_response(example, chunk, response):
    parsed_concepts = []
    
    # Regular expression to match 'Concept', 'Description', and 'Examples' structure
    pattern = r"'Concept':\s*'([^']+)',\s*'Description':\s*'([^']+)',\s*'Examples':\s*\[(.*?)\]"
    
    # Find all matches in the response
    matches = re.findall(pattern, response, re.DOTALL)
    
    print(matches)  # Debugging print to see matched patterns
    
    for concept, description, examples_str in matches:
        print(concept)  # Debugging print to verify concept matching
        # Split examples string into a list, handling commas and quotes
        examples = [ex.strip().strip("'\"") for ex in examples_str.split(',')]
        # Remove empty examples
        examples = [ex for ex in examples if ex]
        
        # Add parsed concept, description, and examples to the results
        parsed_concepts.append({
            'id': example['example_id'],
            'seed': chunk,
            'Concept': concept.strip(),
            'Description': description.strip(),
            'Examples': examples,
        })
    
    return parsed_concepts


async def generate_concepts(client, examples, args):
    concept_responses = []

    for example in examples:
        seed = example["seed"]
        chunks = [seed[i:i+4000] for i in range(0, len(seed), 4000)]

        all_prompts = []
        for chunk in chunks:
            prompt = (
                f"You are an expert in Python fundamentals and programming logics. Given the text snippet below, your task is to identify all key concepts related to python programming that are mentioned. Additionally, provide a short description and locate any examples in the text (either in the form of code snippets or demonstrative one-liner examples) that correspond to each concept. Present the results in the following structured format, using key-value pairs for concepts, concept description and a list of examples corresponding to each concept. A single concept may have multiple examples. Use at most 10 concepts and 10 examples. Do not include anything other than concepts, descriptions, and examples,\n\n"
                f"Generate your response in the following format:\n"
                f"{{\n"
                f"  'Concept': 'Concept1', 'Description': 'Add concept description here', 'Examples': [Example1, Example2, ...],\n"
                f"  'Concept': 'Concept2', 'Description': 'Add concept description here', 'Examples': [Example1, Example2, ...],\n"
                f"  ...\n"
                f"}}\n\n"
                f"Text Snippet:\n\n{chunk}"
            )
            all_prompts.append(prompt)

        for prompt in all_prompts:
            response = client.get_model_response(
                system_prompt="You are a helpful assistant who is expert in python programming.",
                user_prompt=prompt,
                model_id=args.model,
                max_new_tokens=args.max_output_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=1
            )
            # print(response)
            json_block = extract_json_block(response)
            if json_block:
                response = json_block  # Replace response with the extracted JSON block
            print(response)
            parsed_response = parse_response(example,chunk,  response)

            print(parsed_response)
            concept_responses.extend(parsed_response)
    
    return concept_responses



import re



def extract_python_code(response_text):
    """
    Extract Python code from the response text by identifying code blocks
    in the following formats:
    
    1. ```python\n<Your code here>\n```
    2. ```\n<Your code here>\n```
    3. ```<Your code here>```
    
    If none of these formats are matched, include the entire response text directly if it's not empty.
    
    This version handles cases where there are multiple markdown blocks 
    and concatenates them into a single code block.
    """
    # Regex patterns for various code block formats
    patterns = [
        r"```python\n(.*?)\n```",  # Pattern for ```python\n<code>\n```
        r"```\n(.*?)\n```",        # Pattern for ```\n<code>\n```
        r"```(.*?)```"             # Pattern for ```<code>```
    ]
    
    # Initialize a variable to collect all code blocks
    all_code_blocks = []
    
    # Loop through patterns and try to find all matches
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)  # Find all matches for the pattern
        for match in matches:
            all_code_blocks.append(match.strip())  # Collect code blocks and remove any extra whitespace
    
    # Check if any code blocks were found
    if all_code_blocks:
        # Join all code blocks into a single string separated by newlines
        return "\n".join(all_code_blocks).strip()
    
    # If no code block matches, but the response is not empty, return it directly
    if response_text.strip():
        return response_text.strip()
    
    # If no code block is found and the response is empty, return an empty string
    return ""

    
import time  # To add a small delay between retries

async def generate_instruction_response(client, concepts, args, start_id):
    level=2
    if level==1:
        difficulty = ['easy']
    else:
        difficulty = ['easy', 'medium', 'hard']
    
    instruction_regex = re.compile(r"Instruction\d+:\n(.*?)\n", re.DOTALL)
    current_id = start_id 
    
    with open(Path(f"synthesized_pythondocs_instruction_responses_testcases-final.json"), 'a') as f_res:
        for concept in concepts:
            for d in difficulty:
                instruction_prompt = (
                    f"Given the concepts and examples below, generate six distinct instructions with complexity level: {d}.\n"
                    f"Ensure that the tasks are as non-overlapping as possible while covering diverse aspects of the concepts.\n"
                    f"The 'easy' tasks should be straightforward, requiring simple Python programs, while 'medium' tasks should involve some logical complexity.\n"
                    f"The 'hard' tasks should be sophisticated and intricate, designed to result in solutions that span a large number of lines of code.\n\n"
                    
                    f"Concept:\n{concept['Concept']}\n\n"
                    f"Description:\n{concept['Description']}\n\n"
                    f"Examples:\n{', '.join(concept['Examples'])}\n\n"
                    
                    f"Instructions should not contain anything other than just the instruction. Generate your response in the following format:\n"
                    f"Instruction1:\n{{}}\n\n"
                    f"Instruction2:\n{{}}\n\n"
                    f"Instruction3:\n{{}}\n\n"
                    f"Instruction4:\n{{}}\n\n"
                    f"Instruction5:\n{{}}\n\n"
                    f"Instruction6:\n{{}}"
                )

                instruction_response = client.get_model_response(
                    system_prompt="You are a helpful assistant.",
                    user_prompt=instruction_prompt,
                    model_id=args.model,
                    max_new_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    top_k=50,
                    top_p=1
                )
                
                instructions = instruction_regex.findall(instruction_response)
                
                for idx, instruction in enumerate(instructions):
                    response_codes = []
                    code_based_test_cases = []
                    instruction_based_test_cases = []
                    
                    # Generate 5 instruction-based test cases
                    for _ in range(5):
                        instruction_test_prompt = (
                            f"Generate test cases for a Python function that would solve the following programming task. "
                            f"Create test cases based solely on the task description, without seeing any implementation. "
                            f"Include various scenarios, edge cases, and expected behaviors.\n\n"
                            f"Task Description:\n{instruction}\n\n"
                            f"Generate direct assertions that would verify the correct behavior of any implementation.\n"
                            f"Make sure your response contains only code and nothing else.\n\n"
                            f"Always generate the test cases in the following format, using triple backticks to denote a code block:\n"
                            f"```python\n"
                            f"def test_function_requirements():\n"
                            f"    # Basic functionality tests\n"
                            f"    assert ...  # Test case 1 (basic functionality)\n"
                            f"    assert ...  # Test case 2 (edge cases)\n"
                            f"    assert ...  # Test case 3 (expected behavior)\n"
                            f"    assert isinstance(..., ...)  # Type checks if applicable\n"
                            f"```\n"
                        )
                        
                        instruction_test_response = client.get_model_response(
                            system_prompt="You are an expert at python testing and requirements analysis.",
                            user_prompt=instruction_test_prompt,
                            model_id=args.model,
                            max_new_tokens=args.max_output_tokens,
                            temperature=args.temperature,
                            top_k=50,
                            top_p=1
                        )
                        
                        instruction_test_case = extract_python_code(instruction_test_response)
                        if instruction_test_case:
                            instruction_based_test_cases.append(instruction_test_case)
                        else:
                            instruction_based_test_cases.append("")

                    # Generate code solutions and code-based test cases (existing logic)
                    # for _ in range(5):
                        response_prompt = (
                            f"Given the instruction below, your task is to generate functionally correct Python code. The generated code must strictly follow these constraints:\n\n"
                            f"1. The entire code must be enclosed within a single function definition. No part of the code should exist outside of this function.\n"
                            f"2. Ensure the code is fully modular and does not rely on any external code or global variables. All logic and dependencies should be handled within the single function.\n"
                            f"3. The code should be optimized for readability and follow Python best practices.\n\n"
                            f"4. Do not include anything other than the code.\n"
                            f"5. Make sure the code is indented and included. \n"
                            f"Instruction:\n"
                            f"{instruction}\n\n"
                            f"Always Generate the Python code block in the following format, using triple backticks to denote a code block:\n"
                            f"```python\n"
                            f"def function():\n"
                            f"    # Add your function logic here\n"
                            f"\n"
                            f"```"
                        )
                        
                        max_retries = 5
                        retry_count = 0
                        response_response = ""
                        while retry_count < max_retries:
                            response_response = client.get_model_response(
                                system_prompt="You are a Python programming expert.",
                                user_prompt=response_prompt,
                                model_id=args.model,
                                max_new_tokens=args.max_output_tokens,
                                temperature=args.temperature,
                                top_k=50,
                                top_p=0.8
                            )
                            
                            if response_response.strip():
                                break
                            else:
                                retry_count += 1
                                print(f"Retrying... {retry_count}/{max_retries}")
                                time.sleep(1)

                        if not response_response.strip():
                            print(f"Failed to get a valid response after {max_retries} retries.")
                            continue

                        python_code = extract_python_code(response_response)

                        if python_code:
                            test_case_prompt = (
                                f"Generate simple, direct unit test cases for the following Python code. Do not use mocking, nonlocal variables, or any advanced testing techniques. "
                                f"Ensure that the tests cover basic functionality, boundary conditions, and include checks for input and output types if such information is available in the task description.\n\n"
                                f"Task Description and Python Code:\n"
                                f"Task Description:\n{instruction}\n\n"
                                f"```python\n{python_code}\n```\n\n"
                                f"The test cases should be direct assertions without altering the original code structure.\n\n"
                                f"All unit tests should be handled within the single test function.\n\n"
                                f"Make sure your response contains only code and nothing else.\n\n"
                                f"Always generate the test cases in the following format, using triple backticks to denote a code block:\n"
                                f"```python\n"
                                f"def test_function():\n"
                                f"    # Add basic unit tests inside this function\n"
                                f"    assert ...  # Test case 1 (basic functionality)\n"
                                f"    assert ...  # Test case 2 (boundary conditions)\n"
                                f"    assert isinstance(..., ...)  # Test case for input/output type (if available)\n"
                                f"    assert ...  # Additional test cases\n"
                                f"```\n"
                            )

                            test_cases_response = client.get_model_response(
                                system_prompt="You are an expert at python and an expert test case generator.",
                                user_prompt=test_case_prompt,
                                model_id=args.model,
                                max_new_tokens=args.max_output_tokens,
                                temperature=args.temperature,
                                top_k=50,
                                top_p=1
                            )

                            test_case = extract_python_code(test_cases_response)
                            if test_case:
                                code_based_test_cases.append(test_case)
                            else:
                                code_based_test_cases.append("")

                        if python_code:
                            response_codes.append(python_code)
                        else:
                            response_codes.append(response_response)
                    
                    current_id += 1
                    
                    # Store results with both types of test cases
                    result = {
                        "concept_id": concept["id"],
                        "id": current_id,
                        "seed": concept["seed"],
                        "concept": concept["Concept"],
                        "concept_description": concept["Description"],
                        "concept_examples": concept["Examples"],
                        "instruction": instruction,
                        "responses": response_codes,
                        "code_based_tests": code_based_test_cases,
                        "instruction_based_tests": instruction_based_test_cases,
                        "difficulty": d
                    }

                    f_res.write(json.dumps(result) + "\n")
                    f_res.flush()

    return start_id



import os
import json

def load_bug_data(file_path):
    bug_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                bug_info = json.loads(line.strip())
                bug_data.append(bug_info)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")
                continue
    return bug_data

async def main():
    # Parsing arguments
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    concept_file_path = Path(f"evolved-concepts-all-test.json")

    if concept_file_path.exists():
        print(f"Concept file already exists at {concept_file_path}, skipping concept generation.")
        concepts = load_bug_data(concept_file_path)
        print(f"Loaded {len(concepts)} concepts from the input file.")
    else:
        pass

    # Initialize client based on model type
    if args.model_type == "BAM":
        client = GenAIClient()  # Adjust this according to BAM client
    elif args.model_type == "OpenAI":
        client = OpenAIClient()
    elif args.model_type == "HuggingFace":
        client = HuggingFaceClient()  # Adjust this according to Hugging Face client
    elif args.model_type == "vllm":
        client = VLLMClient()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    

    # Check if the output file already exists
    processed_example_ids = set()
    output_file_path = Path(f"synthesized_pythondocs_instruction_responses_testcases-final.json")
    
    if output_file_path.exists():
        # Load the existing file and extract processed example_ids
        print(f"Loading existing results from {output_file_path}")
        with open(output_file_path, 'r') as f_res:
            for line in f_res:
                existing_result = json.loads(line)
                processed_example_ids.add(existing_result["concept_id"])

    
    # Filter the dataset to exclude already processed examples
    unprocessed_concepts = [example for example in concepts if example["id"] not in processed_example_ids]
    # print(len(unprocessed_dataset))
    if not unprocessed_concepts:
        print("All examples are already processed.")
        return
    

    # Generate instructions/responses if instruct_mode is C->I or I->R
    print(f"Generating instructions/responses and saving to {output_file_path}")
    start_id = max(processed_example_ids) + 1 if processed_example_ids else 0  # Set initial ID

    chunked_concepts = list(star_align.utils.chunked(unprocessed_concepts, n=args.num_batched_requests))
    pbar = tqdm(chunked_concepts, desc="Processing Seed Examples")
    print(f"Number of unprocessed concepts: {len(unprocessed_concepts)}")
    print(f"Number of chunks: {len(chunked_concepts)}")
    print(f"Chunk size: {args.num_batched_requests}")
    for concepts in pbar:
        start_id = await generate_instruction_response(client, concepts, args, start_id)
        
    

if __name__ == "__main__":
    asyncio.run(main())
