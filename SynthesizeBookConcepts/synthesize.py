import os
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
import sys
import time
import asyncio
from tqdm import tqdm


## USAGE
# python selfcodealign/src/star_align/synthesize_ICT_v2.py --instruct_mode "C->I" --seed_data concepts.json --model model_id --model_type vllm
sys.path.append('/proj/data-eng/users/shailja/SynthesizeBookProgConcepts/selfcodealign/src/')
import star_align

InstructMode = Literal["I->R", "S->C", "C->I"]

@dataclass(frozen=True)
class Args:
    seed_data_file: str
    model: str
    instruct_mode: InstructMode
    model_type: str
    seed_code_start_index: int = 0
    max_new_data: int = 10
    continue_from: str | None = None
    seed: int = 3407
    temperature: float = 0.7
    max_output_tokens: int = 1536
    prompting_mode: Literal["chat", "completion"] = "completion"
    num_batched_requests: int = 1
    num_seed_examples: int = 10
    num_sample_per_request: int = 32
    sleep: float | None = None
    delay: float | None = None
    tag: str = ""
    save_dir: str = "./"

    def fingerprint(self) -> str:
        args = (self.seed_data_file, self.seed, self.prompting_mode, 
                self.temperature, self.model, self.max_output_tokens)
        return star_align.utils.compute_fingerprint(*args, hash_length=5)

def load_prompts(prompt_file="prompts.json"):
    try:
        with open(prompt_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found. Please ensure it exists in the working directory.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in '{prompt_file}'. Please check the file content.")

def extract_json_block(response):
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, response, re.DOTALL)
    return match.group(1) if match else None

def parse_response(example, chunk, response):
    parsed_concepts = []
    pattern = r"'Concept':\s*'([^']+)',\s*'Description':\s*'([^']+)',\s*'Examples':\s*\[(.*?)\]"
    matches = re.findall(pattern, response, re.DOTALL)
    
    for concept, description, examples_str in matches:
        examples = [ex.strip().strip("'\"") for ex in examples_str.split(',') if ex.strip()]
        parsed_concepts.append({
            'id': example['example_id'],
            'seed': chunk,
            'Concept': concept.strip(),
            'Description': description.strip(),
            'Examples': examples,
        })
    return parsed_concepts

async def generate_concepts(client, examples, args, prompts):
    concept_responses = []
    for example in examples:
        seed = example["seed"]
        chunks = [seed[i:i+4000] for i in range(0, len(seed), 4000)]
        
        for chunk in chunks:
            prompt = prompts["generate_concepts_prompt"].format(chunk=chunk)
            response = client.get_model_response(
                system_prompt="You are a helpful assistant who is an expert in Python programming.",
                user_prompt=prompt,
                model_id=args.model,
                max_new_tokens=args.max_output_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=1
            )
            json_block = extract_json_block(response)
            if json_block:
                response = json_block
            concept_responses.extend(parse_response(example, chunk, response))
    return concept_responses

def get_function_signature(code):
    match = re.search(r'^def\s+\w+\s*\(.*?\):', code, re.MULTILINE)
    return match.group(0) if match else ""

def get_signature_details(code):
    """
    Extracts signature details from Python code, handling both standalone functions and classes.
    Returns a dictionary with:
    - 'type': 'function' or 'class'
    - 'name': function or class name
    - 'signature': full signature string (for functions) or None (for classes)
    - 'methods': list of method signatures (for classes) or None (for functions)
    - Each signature includes input parameters and assumes a return value.
    """
    code = code.strip()
    
    # Check if it's a class
    class_match = re.search(r'^class\s+(\w+)\s*(?:\([^)]*\))?:', code, re.MULTILINE)
    if class_match:
        class_name = class_match.group(1)
        # Extract all method signatures within the class
        method_pattern = r'^\s{4,}def\s+(\w+)\s*\((.*?)\)\s*:'
        methods = re.findall(method_pattern, code, re.MULTILINE)
        method_signatures = [
            f"def {method_name}({params}): # Returns a value" for method_name, params in methods
        ]
        return {
            'type': 'class',
            'name': class_name,
            'signature': None,
            'methods': method_signatures if method_signatures else ["# No methods found"]
        }
    
    # Assume standalone function
    func_match = re.search(r'^def\s+(\w+)\s*\((.*?)\)\s*:', code, re.MULTILINE)
    if func_match:
        func_name = func_match.group(1)
        params = func_match.group(2)
        return {
            'type': 'function',
            'name': func_name,
            'signature': f"def {func_name}({params}): # Returns a value",
            'methods': None
        }
    
    return {
        'type': 'unknown',
        'name': '',
        'signature': '# No valid signature found',
        'methods': None
    }


def extract_python_code(response_text):
    patterns = [
        r"```python\n(.*?)\n```",
        r"```\n(.*?)\n```",
        r"```(.*?)```"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return matches[0].strip()
    return response_text.strip() if response_text.strip() else ""

def split_test_cases(test_code):
    test_functions = re.findall(r'(def test_[^(]+\([^)]*\):(?:\n\s+[^\n]+)*)', test_code, re.MULTILINE)
    return [func.strip() for func in test_functions if func.strip()]

async def generate_instruction_response(client, concepts, args, prompts):
    difficulty_levels = ['easy', 'medium', 'hard']
    instruction_regex = re.compile(r"Instruction\d+:\n(.*?)\n", re.DOTALL)
    current_id = args.seed_code_start_index
    min_score = 6 # Filter for complex, novel concepts
    
    difficulty_desc = {
        'easy': {
            'complexity_description': 'straightforward problems requiring simple Python solutions (1-3 difficulty)',
            'expected_lines': '10-30 lines',
            'difficulty_score': '1-3 on a scale of 1-10'
        },
        'medium': {
            'complexity_description': 'moderately complex problems requiring algorithmic thinking (5-7 difficulty)',
            'expected_lines': '30-60 lines',
            'difficulty_score': '5-7 on a scale of 1-10'
        },
        'hard': {
            'complexity_description': 'sophisticated problems requiring complex algorithms and data structures (8-10 difficulty)',
            'expected_lines': '50-100+ lines',
            'difficulty_score': '8-10 on a scale of 1-10'
        }
    }
    
    with open(Path("synthesized_ICT_v1.json"), 'a') as f_res:
        for concept in concepts:
            #if concept.get('score', 0) < min_score:
            #    print(f"Skipping {concept['Concept']} (score: {concept.get('score', 0)}) - below minimum score {min_score}")
            #    continue
            
            for d in difficulty_levels:
                diff_config = difficulty_desc[d]

                instruction_prompt = prompts["instruction_prompt"].format(
                    difficulty=d,
                    complexity_description=diff_config['complexity_description'],
                    expected_lines=diff_config['expected_lines'],
                    difficulty_score=diff_config['difficulty_score'],
                    concept=concept['Concept'],
                    description=concept['Description'],
                    examples=', '.join(concept['Examples'])
                )
                instruction_response = client.get_model_response(
                    system_prompt="You are a helpful assistant with expertise in Python and instructional design.",
                    user_prompt=instruction_prompt,
                    model_id=args.model,
                    max_new_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    top_k=50,
                    top_p=0.8
                )
                instructions = instruction_regex.findall(instruction_response)
                
                for instruction in instructions:
                    response_codes = []  # To store five code responses
                    instruction_based_test_cases = []  # To store test cases for each code response
                    
                    # Generate five code solutions
                    for _ in range(5):
                        code_prompt = prompts["code_prompt"].format(
                            difficulty=d,
                            complexity_description=diff_config['complexity_description'],
                            expected_lines=diff_config['expected_lines'],
                            difficulty_score=diff_config['difficulty_score'],
                            instruction=instruction
                        )
                        max_retries = 5
                        for attempt in range(max_retries):
                            code_response = client.get_model_response(
                                system_prompt="You are a Python programming expert.",
                                user_prompt=code_prompt,
                                model_id=args.model,
                                max_new_tokens=args.max_output_tokens,
                                temperature=args.temperature,
                                top_k=50,
                                top_p=0.8
                            )
                            python_code = extract_python_code(code_response)
                            if python_code:
                                break
                            time.sleep(1)
                        else:
                            print(f"Failed to generate valid code for instruction after {max_retries} retries: {instruction}")
                            continue

                        # Get signature details
                        signature_info = get_signature_details(python_code)
                        if signature_info['type'] == 'class':
                            # For classes, assume 'primary_method' or first method as primary
                            method_name = signature_info['methods'][0].split()[1].split('(')[0] if signature_info['methods'] else 'primary_method'
                            test_prompt = prompts["test_prompt"].format(
                                difficulty=d,
                                complexity_description=diff_config['complexity_description'],
                                expected_lines=diff_config['expected_lines'],
                                difficulty_score=diff_config['difficulty_score'],
                                instruction=instruction,
                                function_name=signature_info['name'],  # Class name
                                class_name=signature_info['name'],
                                method_name=method_name,
                                function_signature='\n'.join(signature_info['methods']) if signature_info['methods'] else '# No methods found'
                            )
                        else:
                            # For standalone functions
                            test_prompt = prompts["test_prompt"].format(
                                difficulty=d,
                                complexity_description=diff_config['complexity_description'],
                                expected_lines=diff_config['expected_lines'],
                                difficulty_score=diff_config['difficulty_score'],
                                instruction=instruction,
                                function_name=signature_info['name'],
                                class_name='',  # Empty for functions
                                method_name='',  # Empty for functions
                                function_signature=signature_info['signature']
                            )

                        test_response = client.get_model_response(
                            system_prompt="You are an expert at Python testing and requirements analysis.",
                            user_prompt=test_prompt,
                            model_id=args.model,
                            max_new_tokens=args.max_output_tokens,
                            temperature=args.temperature,
                            top_k=50,
                            top_p=1
                        )
                        test_cases = split_test_cases(extract_python_code(test_response))
                        instruction_based_test_cases.append(test_cases if test_cases else ["# No valid tests generated"])

                        response_codes.append(python_code)

                    current_id += 1
                    
                    result = {
                        "concept_id": concept["id"],
                        "id": current_id,
                        "seed": concept["seed"],
                        "concept": concept["Concept"],
                        "concept_description": concept["Description"],
                        "concept_examples": concept["Examples"],
                        "instruction": instruction,
                        "responses": response_codes,
                        "instruction_based_tests": instruction_based_test_cases,
                        "difficulty": d
                    }
                    f_res.write(json.dumps(result) + "\n")
                    f_res.flush()
    return current_id




def load_bug_data(file_path):
    bug_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                bug_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return bug_data

async def main():
    from transformers import HfArgumentParser
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    concept_file_path = Path(args.seed_data_file)
    
    if not concept_file_path.exists():
        raise FileNotFoundError("Concept file 'evolved-concepts-all-test.json' not found.")
    
    concepts = load_bug_data(concept_file_path)
    
    # Load prompts with error handling
    prompts = load_prompts("prompts.json")

    # Client initialization
    if args.model_type == "BAM":
        client = star_align.utils.GenAIClient()
    elif args.model_type == "OpenAI":
        client = star_align.utils.OpenAIClient()
    elif args.model_type == "HuggingFace":
        client = star_align.utils.HuggingFaceClient()
    elif args.model_type == "vllm":
        client = star_align.utils.VLLMClient()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    output_file_path = Path("synthesized_ICT_v1.json")
    processed_concepts = set()
    if output_file_path.exists():
        with open(output_file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_concepts.add(data["concept"])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {output_file_path}: {line.strip()}")
                    continue
    
    unprocessed_concepts = [c for c in concepts if c["Concept"] not in processed_concepts]
    if not unprocessed_concepts:
        print("All concepts have been synthesized.")
        return
    
    print(f"Found {len(unprocessed_concepts)} unsynthesized concepts to process.")
    with tqdm(total=len(unprocessed_concepts), desc="Synthesizing Concepts") as pbar:
        await generate_instruction_response(client, unprocessed_concepts, args, prompts)
        pbar.update(len(unprocessed_concepts))

if __name__ == "__main__":
    asyncio.run(main())
