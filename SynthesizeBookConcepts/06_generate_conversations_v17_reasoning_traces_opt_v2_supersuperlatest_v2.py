import os
import re
import json
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import VLLMClient_mistral_batch, VLLMClient_mistral


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




# Extract the first function from code file
def extract_first_function(code_directory, key, response_num, testcase_num):
    code_filename = f"{code_directory}/temp_script_{key}_response_{response_num}_testcase_{testcase_num}.py"
    logger.debug(f"Attempting to read code file: {code_filename}")
    try:
        with open(code_filename, 'r') as file:
            full_content = file.read()
        matches = list(re.finditer(r"##################CODE####################", full_content))
        if len(matches) >= 2:
            return full_content[matches[0].end():matches[1].start()].strip()
        elif len(matches) == 1:
            return full_content[:matches[0].start()].strip()  # Everything before the marker
        logger.warning(f"No CODE markers found in {code_filename}")
        return full_content.strip()  # Default to full content if no markers
    except FileNotFoundError:
        logger.error(f"File not found: {code_filename}")
        return None
    except Exception as e:
        logger.error(f"Error reading {code_filename}: {str(e)}")
        return None

# Read raw trace
def read_raw_trace(raw_trace_directory, filename):
    trace_filename = os.path.join(raw_trace_directory, filename)
    logger.debug(f"Attempting to read trace file: {trace_filename}")
    try:
        with open(trace_filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Raw trace file not found: {trace_filename}")
        return None
    except Exception as e:
        logger.error(f"Error reading {trace_filename}: {str(e)}")
        return None

# Read test case content
def read_test(code_directory, key, response_num, testcase_num):
    code_filename = f"{code_directory}/temp_script_{key}_response_{response_num}_testcase_{testcase_num}.py"
    logger.debug(f"Attempting to read test file: {code_filename}")
    try:
        with open(code_filename, 'r', encoding='utf-8') as file:
            content = file.read()
        matches = list(re.finditer(r"##################CODE####################", content))
        if len(matches) >= 2:
            return content[matches[1].end():].strip()
        elif len(matches) == 1:
            return content[matches[0].end():].strip()  # Everything after the marker
        logger.warning(f"No CODE markers found in {code_filename}")
        return None  # No test case if no markers
    except FileNotFoundError:
        logger.error(f"Test file not found: {code_filename}")
        return None
    except Exception as e:
        logger.error(f"Error reading {code_filename}: {str(e)}")
        return None

# Sanitize text
def sanitize_text(text):
    replacements = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '--'
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text

# Process one test case at a time
def process_test_case(vllm_client, key, response_num, testcase_num, filename, code_directory, raw_trace_directory, data_lookup, processed_ids):
    conv_id = f"{key}_{response_num}_{testcase_num}"
    if conv_id in processed_ids:
        logger.debug(f"Skipping already processed {conv_id}")
        return None

    instruction = data_lookup[key]["instruction"]
    entrypoint = data_lookup[key]["entry_point"]
    if not all([instruction, entrypoint]):
        logger.warning(f"Missing instruction or entrypoint for key {key}")
        return None
    
    first_function = extract_first_function(code_directory, key, response_num, testcase_num)
    if not first_function:
        logger.warning(f"Could not extract function for {key}, response {response_num}")
        return None
    
    first_function = '\n'.join(line for line in first_function.split('\n') if not line.strip().startswith('@'))
    instruction_statement = instruction.replace("Explain how to", "This code").replace("number", "")
    if len(instruction_statement.split()) > len(instruction.split()):
        instruction_statement = " ".join(instruction_statement.split()[:len(instruction.split())])

    raw_trace = read_raw_trace(raw_trace_directory, filename)
    if raw_trace == "Execution timed out after 10 seconds":
        logger.info(f"Skipping {conv_id} due to execution timeout in trace")
        return None



    test_case_content = read_test(code_directory, key, response_num, testcase_num)
    if not (raw_trace and test_case_content):
        logger.warning(f"Skipping {conv_id} due to missing trace or test case")
        return None

    # Precompute all required values
    param_prompt = f"""
    You are an expert in Python code analysis. Given a Python function and its test code, identify its input parameter(s) and output parameter. The input is the parameter(s) in the function definition. The output is the variable assigned to the return value, or if there's no return or print, the parameter modified by the function. If you can’t determine either from the function alone, use the test code to infer them.

Function Code: ```python
{first_function}

Test Code: ```python
{test_case_content}

Output Format:
Input: [Input parameter name(s)]
Output: [Output variable name]
"""
    param_resp = vllm_client.get_model_response(
        "You are an expert in Python code analysis.", param_prompt)
    try:
        input_match = re.search(r"Input:\s*\[(.*?)\]", param_resp, re.DOTALL) if param_resp else None
        output_match = re.search(r"Output:\s*\[(.*?)\]", param_resp, re.DOTALL) if param_resp else None
        input_param = input_match.group(1).strip() if input_match else "input"
        output_param = output_match.group(1).strip() if output_match else "result"
    except AttributeError:
        logger.warning(f"Skipping {conv_id} due to malformed parameter response")
        return None

    io_prompt = f"""
You are an expert in code execution analysis. Given a Python function or class, its test case content, determine the input(s) provided and the final expected output produced.
In case when the input and output are explicitly mentiond in the assert statement, use the provided values. 
Else if, the input is explicitly mentioned in the test case content but the output is not, check whether the output is a True/False based on the assert statement,or whether teh output is expected to be a exception or not.
Else if, the input is not explicitly mentioned in the test case content, use the execution trace to identify teh input and output.
Code: ```python
{first_function}
Test Case Content: ```
{test_case_content}
Raw Trace: ```
{raw_trace}
Output Format:
<Input>[Extracted input(s)]</Input>
<Output>[Extracted output]</Output>
"""
    io_resp = vllm_client.get_model_response(
        "You are an expert in code execution analysis.", io_prompt)
    
    try:
        input_match = re.search(r"<Input>(.*?)</Input>", io_resp, re.DOTALL)
        input_val = input_match.group(1).strip() if input_match else "unknown input"
        output_match = re.search(r"<Output>(.*?)</Output>", io_resp, re.DOTALL)
        output_val = output_match.group(1).strip() if output_match else "unknown output"
    except AttributeError:
        logger.warning(f"Skipping {conv_id} due to malformed input/output response")
        return None

    #input_val = re.search(r"<Input>(.*?)</Input>", io_resp, re.DOTALL).group(1).strip() if io_resp else "unknown input"
    #output_val = re.search(r"<Output>(.*?)</Output>", io_resp, re.DOTALL).group(1).strip() if io_resp else "unknown output"
    
    if input_val == "unknown input":
        input_matches = re.findall(rf"{entrypoint}\((.*?)\)", test_case_content)
        input_val = input_matches[0] if input_matches else "None"
    if output_val == "unknown output":
        try:
            output_match = re.search(r"Return value:.. (.*)", raw_trace)
            output_val = output_match.group(1).strip() if output_match else "unknown output"
        except AttributeError:
            logger.warning(f"Skipping {conv_id} due to malformed trace return value")
            return None
    # Prepare all five prompts with precomputed values
    system_prompts = [
        "You are an expert at crafting natural questions about code.",
        "You are an expert in code execution analysis who explains things clearly and naturally.",
        "You are an expert in code execution analysis who reflects naturally in third-person.",
        "You are an expert in code execution analysis who explains things clearly and naturally.",
        "You are an expert in code execution analysis who reflects naturally in third-person."
    ]
    user_prompts = [
        # Questions (unchanged)
        f"""
You are an expert at crafting natural, varied questions about code execution. Given a Python function, its entry point, and an input and output, generate two concise questions.
- A forward question to identify the output for the given input.
- A backward question to identify the expected input for the given output.
You will never refuse to generate the two questions, and you will always generate the two questions in the output format.

Code: ```python
{first_function}
Entry Point: {entrypoint}
Given Input: {input_val}
Given Output: {output_val}
Output Format:
<ForwardQuestion>[Your forward question]</ForwardQuestion>
<BackwardQuestion>[Your backward question]</BackwardQuestion>
""",
        # Forward Reasoning (added test case guideline)
        f"""
You are an expert in code execution who explains things like a patient teacher. Given a Python function, its raw execution trace for the given input and output,, and provided input/output values, explain how the input leads to the output. Use the raw trace as evidence to guide your reasoning steps and ensure alignment and accuracy, presenting your explanation as a clear, deductive sequence of steps—starting with the code’s structure and logic, then logically deducing the outcome. If there are loops, don’t walk through each iteration; instead, summarize their overall effect and connections like a human would. Mention the predicted output naturally near the end of your steps, after describing the key operations, not at the start.
Code: ```python
{first_function}
Raw Trace: ```
{raw_trace}
Given Input: {input_val}
Given Output: {output_val}
Output Format:
<Steps>
1. [Natural first-person step]
2. [Next step]
</Steps>
{{output}}{output_val}{{/output}}
""",
        # Forward Feedback (unchanged)
        f"""
Reflect on this forward reasoning for the code:
```python
{first_function}
Given Input: {input_val}
Expected Output (from trace): {output_val}
Predicted Output (from reasoning): {{predicted_output_placeholder}}
Trace: ```
{raw_trace}
Use third-person narration to compare the predicted output to the expected output.
""",
        # Backward Reasoning (added test case guidelines)
        f"""
You are an expert in code execution, reasoning like a detective unraveling a mystery in reverse. Given a Python function, its raw execution trace, and provided input/output values, deduce the most plausible input(s) that produce the given output. Begin with the output and the final line of the code, working *backward* through the code and execution trace to reconstruct state transitions and variable changes leading to the input. Use the execution trace as evidence to ground your reasoning and ensure accuracy. Your explanation must be a clear, step-by-step deduction, starting with the output and reversing through the code’s logic to deduce the input(s). Do *not* mention or assume any input until the final step, focusing each step on the code’s operations, trace evidence, and logical conditions without jumping to conclusions. Use the trace to guide your steps, but do not rely on any input value upfront to shortcut the process. For loops or repetitive structures, summarize their overall effect and connections, as a human would reason. If multiple inputs could produce the same output based on the code and trace, list all equally likely possibilities in the final step, justifying each with the trace and logic. For example, if a function adds two numbers to produce an output of 6.0, the trace might show inputs like 2.5 and 3.5, but other pairs like 1.0 and 5.0 or 4.0 and 2.0 could also yield 6.0, depending on the code’s logic; all such valid combinations should be noted. Present the deduced input(s) naturally at the end, after fully exploring the logic and state changes.

Code: ```python
{first_function}

Raw Trace: ```
{raw_trace}

Given Input: {input_val}
Given Output: {output_val}
Output Format:
<Steps>
1. [Begin with the output and final code line, analyzing their relationship and trace evidence]

2. [Continue backward, examining prior code and trace to deduce state changes]

3. [Further reverse step, connecting logic and conditions]
...
N. [Conclude with the deduced input(s), tying together the reverse analysis]
</Steps>
{{input}}
For a single input, list it directly.
For multiple inputs, list each on a new line as "Plausible input 1: <value>", "Plausible input 2: <value>", etc.
{input_val}
{{/input}}
        """,

        # Backward Feedback (unchanged)
        f"""
Reflect on this backward reasoning for the code:
```python
{first_function}
Expected Input (from trace): {input_val}
Given Output: {output_val}
Predicted Input (from reasoning): {{predicted_input_placeholder}}
Trace: ```
{raw_trace}
Use third-person narration to compare the predicted input to the expected input.
"""
    ]

    # First batch: Questions, Forward Reasoning, Backward Reasoning
    first_batch_systems = [system_prompts[0], system_prompts[1], system_prompts[3]]
    first_batch_prompts = [user_prompts[0], user_prompts[1], user_prompts[3]]
    first_batch_responses = vllm_client.get_model_response(first_batch_systems, first_batch_prompts)
    if not first_batch_responses or any(r is None for r in first_batch_responses):
        logger.warning(f"LLM failed for {conv_id} in first batch")
        return None
    print(user_prompts[3])
    questions_resp, forward_resp, backward_resp = first_batch_responses

    # Parse questions
    try:
        forward_match = re.search(r"<ForwardQuestion>(.*?)</ForwardQuestion>", questions_resp, re.DOTALL)
        forward_q = forward_match.group(1).strip() if forward_match else f"What does `{entrypoint}` return for {input_val}? Show the steps."
        backward_match = re.search(r"<BackwardQuestion>(.*?)</BackwardQuestion>", questions_resp, re.DOTALL)
        backward_q = backward_match.group(1).strip() if backward_match else f"What input to `{entrypoint}` could give {output_val}? Explain how."
    except AttributeError:
        logger.warning(f"Skipping {conv_id} due to malformed questions response")
        return None

    # Extract predicted values from reasoning
    try:
        forward_output_match = re.search(r"\{output\}(.*?)\{/output\}", forward_resp, re.DOTALL)
        predicted_output = forward_output_match.group(1).strip() if forward_output_match else "unknown"
    except AttributeError:
        logger.warning(f"Skipping {conv_id} due to malformed forward response")
        return None

    try:
        backward_input_match = re.search(r"\{input\}(.*?)\{/input\}", backward_resp, re.DOTALL)
        predicted_input = backward_input_match.group(1).strip() if backward_input_match else "unknown"
    except AttributeError:
        logger.warning(f"Skipping {conv_id} due to malformed backward response")
        return None

    # Second batch: Forward Feedback, Backward Feedback with pre-substituted placeholders
    forward_feedback_prompt = user_prompts[2].replace("{{predicted_output_placeholder}}", predicted_output)
    backward_feedback_prompt = user_prompts[4].replace("{{predicted_input_placeholder}}", predicted_input)
    second_batch_systems = [system_prompts[2], system_prompts[4]]
    second_batch_prompts = [forward_feedback_prompt, backward_feedback_prompt]
    second_batch_responses = vllm_client.get_model_response(second_batch_systems, second_batch_prompts)
    if not second_batch_responses or any(r is None for r in second_batch_responses):
        logger.warning(f"LLM failed for {conv_id} in second batch")
        return None

    forward_feedback_resp, backward_feedback_resp = second_batch_responses


    # Construct conversation, conditionally include parameters and tag outputs
    params_text = f"Input Parameters: {input_param}\nOutput Parameters: {output_param}\n" if input_match and output_match else ""
    conversation = {
        "id": conv_id,
        "instruction": instruction,
        "code": first_function,
        "test_cases": {str(testcase_num): test_case_content},
        "messages": [
            {"content": sanitize_text(f"{instruction_statement}\nHere's the code:\n```python\n{first_function}\n{params_text}{forward_q}"), "role": "user"},
            {"content": sanitize_text(f"{forward_resp}"), "role": "assistant"},
            {"content": sanitize_text(forward_feedback_resp), "role": "assistant"},
            {"content": sanitize_text(backward_q), "role": "user"},
            {"content": sanitize_text(f"{backward_resp}"), "role": "assistant"},
            {"content": sanitize_text(backward_feedback_resp), "role": "assistant"}
        ]
    }

    print(conversation)
    return conversation

# Main execution
data_dir = "data"
code_directory = os.path.join(data_dir, "test_code_1")
raw_trace_directory = os.path.abspath("data/pysnooper_trace_cleaned_v1")
logger.info(f"Starting script with code_dir={code_directory}, trace_dir={raw_trace_directory}")

vllm_client = VLLMClient_mistral_batch()

with open(os.path.join(data_dir, "test_entry_points.json"), "r") as f:
    data = [json.loads(line) for line in f]
data_lookup = {str(entry["task_id"]): {"instruction": entry["instruction"], "entry_point": entry.get("entry_point")} 
               for entry in data if "task_id" in entry}
logger.info(f"Loaded {len(data_lookup)} entries into lookup")

processed_ids = set()
output_file = 'conversations_opt_1_Q7B.jsonl'
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        for line in f:
            try:
                conv = json.loads(line.strip())
                processed_ids.add(conv["id"])
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {output_file}")

trace_files = {}
if os.path.exists(raw_trace_directory):
    for filename in os.listdir(raw_trace_directory):
        match = re.match(r"trace_(\d+)_response_(\d+)_testcase_(\d+)\.(log|txt)", filename)
        if match:
            key, response_num, testcase_num = match.groups()[:-1]
            trace_files[(key, response_num, testcase_num)] = filename

conversations = []
with open(output_file, "a") as file:
    for (key, response_num, testcase_num), filename in sorted(trace_files.items()):
        logger.info(f"Processing {key}_{response_num}_{testcase_num}")
        conversation = process_test_case(
            vllm_client, key, response_num, testcase_num, filename,
            code_directory, raw_trace_directory, data_lookup, processed_ids
        )
        if conversation:
            json.dump(conversation, file)
            file.write("\n")
            file.flush()
            conversations.append(conversation)
            processed_ids.add(conversation["id"])
            logger.debug(f"Saved conversation for {conversation['id']}")

logger.info(f"Generated {len(conversations)} conversations total")
