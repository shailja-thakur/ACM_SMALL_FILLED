from tqdm import tqdm
import json
import sys
import spacy
import pytextrank
from typing import List, Dict
#sys.path.append("/proj/data-eng/users/shailja/SynthesizeBookConcepts/selfcodealign/src/")
sys.path.append("/proj/data-eng/users/shailja/SynthesizeBookConcepts")
from utils import VLLMClient
#import star_align
import re
# HuggingFace model client initialization
client = VLLMClient()
#model_id = "mistralai/mixtral-8x7b-instruct-v01"
model_id="Qwen/Qwen2.5-Coder-7B-Instruct"
# Load the spaCy model and add PyTextRank to the pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Load the JSON file (replace with the actual path to your file)
json_file = "text_chunks_v1.json"
with open(json_file, "r") as f:
    data = json.load(f)

def extract_json_block(response):
    # Match the pseudo-JSON block within triple backticks or similar markers
    json_block_pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, response, re.DOTALL)
    
    if match:
        return match.group(1)  # Return the matched JSON-like block
    return None
 


def parse_response(example: Dict, chunk: str, response: str, topic: str, subtopic: str) -> List[Dict]:
    """
    Parses a model response to extract complete JSON objects for concepts, descriptions, and examples.
    """
    parsed_concepts = []
    
    # Preprocess response to fix common JSON issues
    def preprocess_response(response: str) -> str:
        response = re.sub(r"//.*", "", response)  # Remove inline comments
        response = re.sub(r",\s*([\]}])", r"\1", response)  # Remove trailing commas
        return response
    
    response = preprocess_response(response)
    
    # Updated regex pattern to match JSON-like structures without recursive pattern
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    # Find all JSON blocks in the response
    json_blocks = re.findall(json_pattern, response, re.DOTALL)

    for block in json_blocks:
        try:
            # Attempt to parse each JSON block
            concept_data = json.loads(block)

            # Extract fields if they exist
            concept = concept_data.get("Concept", "").strip()
            description = concept_data.get("Description", "").strip()
            examples = concept_data.get("Examples", [])
            
            # Ensure examples are properly formatted
            if isinstance(examples, str):
                # Handle case where examples might be a string
                examples = [examples.strip()]
            else:
                examples = [ex.strip() for ex in examples if ex and isinstance(ex, str)]

            # Append to parsed concepts
            parsed_concepts.append({
                'seed': chunk,
                'topic': topic,
                'subtopic': subtopic,
                'concept': concept,
                'description': description,
                'examples': examples,
            })
        except json.JSONDecodeError as e:
            # If parsing fails, log the issue
            print(f"Skipping invalid JSON block: {block}\nError: {e}")

    return parsed_concepts




def parse_response_v3(example: Dict, chunk: str, response: str, topic: str, subtopic: str) -> List[Dict]:
    """
    Parses a model response to extract complete JSON objects for concepts, descriptions, and examples.
    """
    parsed_concepts = []
    
    # Preprocess response to fix common JSON issues
    def preprocess_response(response: str) -> str:
        response = re.sub(r"//.*", "", response)  # Remove inline comments
        response = re.sub(r",\s*([\]}])", r"\1", response)  # Remove trailing commas
        return response
    
    response = preprocess_response(response)
    
    # Improved regex for JSON blocks
    json_pattern = r"\{(?:[^{}]|(?R))*\}"

    # Find all JSON blocks in the response
    json_blocks = re.findall(json_pattern, response, re.DOTALL)

    for block in json_blocks:
        try:
            # Attempt to parse each JSON block
            concept_data = json.loads(block)

            # Extract fields if they exist
            concept = concept_data.get("Concept", "").strip()
            description = concept_data.get("Description", "").strip()
            examples = concept_data.get("Examples", [])
            
            # Ensure examples are properly formatted
            examples = [ex.strip() for ex in examples if ex.strip()]

            # Append to parsed concepts
            parsed_concepts.append({
                'seed': chunk,
                'topic': topic,
                'subtopic': subtopic,
                'concept': concept,
                'description': description,
                'examples': examples,
            })
        except json.JSONDecodeError as e:
            # If parsing fails, log the issue
            print(f"Skipping invalid JSON block: {block}\nError: {e}")

    return parsed_concepts


def parse_response_old_v2(example: Dict, chunk: str, response: str, topic: str, subtopic: str) -> List[Dict]:
    """
    Parses a model response to extract complete JSON objects for concepts, descriptions, and examples.

    Parameters:
    - example: Metadata about the example being processed.
    - chunk: The input text chunk.
    - response: The raw response string from the model.
    - topic: The main topic associated with the chunk.
    - subtopic: The subtopic associated with the chunk.

    Returns:
    - A list of dictionaries containing parsed concepts, descriptions, and examples.
    """
    parsed_concepts = []

    # Regular expression to extract JSON-like blocks
    json_pattern = r"\{.*?\}"  # Matches any complete JSON block

    # Find all JSON blocks in the response
    json_blocks = re.findall(json_pattern, response, re.DOTALL)
    #print('EXAMPLE',example)
    for block in json_blocks:
        try:
            # Attempt to parse each JSON block
            concept_data = json.loads(block)

            # Extract fields if they exist
            concept = concept_data.get("Concept", "").strip()
            description = concept_data.get("Description", "").strip()
            examples = concept_data.get("Examples", [])
            #print('concept',concept)
            #print('description',description)
            #print('examples',examples)
            # Ensure examples are properly formatted
            examples = [ex.strip() for ex in examples if ex.strip()]

            # Append to parsed concepts
            parsed_concepts.append({
                #'id': example['example_id'],
                'seed': chunk,
                'topic': topic,
                'subtopic': subtopic,
                'concept': concept,
                'description': description,
                'examples': examples,
            })
        except json.JSONDecodeError:
            # If parsing fails, skip the block
            print(f"Skipping invalid JSON block: {block}")

    return parsed_concepts

def parse_response_old(example, chunk, response, topic,subtopic):
    parsed_concepts = []
    
    # Regular expression to match 'Concept: Concept1', Description: '...', Examples: [...] structure
    pattern = r"'Concept:\s*([^']+)',\s*Description:\s*'([^']+)',\s*Examples:\s*\[(.*?)\]"
    
    # Find all matches in the response
    matches = re.findall(pattern, response, re.DOTALL)
    #print(matches)
    for concept, description, examples_str in matches:
        #print('concept',concept)
        #print('description',description)
        #print('examples',examples)

        # Split examples string into a list
        examples = [ex.strip().strip("'\"") for ex in examples_str.split(',')]
        # Remove empty examples
        examples = [ex for ex in examples if ex]
        
        # if examples:  # Only add if there are non-empty examples
        parsed_concepts.append({
            'id':example['example_id'],
            'seed': chunk,
            'topic': topic,
            'subtopic': subtopic,
            'concept': concept.strip(),
            'description': description.strip(),
            'examples': examples,
            
        })
    
    return parsed_concepts


def deduplicate_concepts(concepts):
    """Remove redundant concepts while preserving uniqueness."""
    unique_concepts = []
    seen = set()
    
    for concept in concepts:
        # Create a normalized key for comparison
        normalized_key = concept['Concept'].lower().strip()
        
        if normalized_key not in seen:
            seen.add(normalized_key)
            unique_concepts.append(concept)
    
    return unique_concepts


# Function to extract keywords along with their rank and count
def extract_keywords(text):
    doc = nlp(text)
    keywords = []
    for phrase in doc._.phrases:
        keywords.append({
            'text': phrase.text,
            'rank': phrase.rank,
            'count': phrase.count,
            'chunks': [chunk.text for chunk in phrase.chunks]
        })
    return keywords

def filter_and_refine_keywords(keywords: List[str], text: str) -> List[str]:
    filter_prompt = (
        f"Review these extracted keywords: {keywords}\n"
        f"1. Filter out any keyword texts unrelated to Python programming.\n"
        f"2. Identify keywords that contain special characters or appear as incomplete sentences. Extract valid Python-related concepts from them if possible.\n"
        f"3. Remove any keywords that remain invalid or incomplete after refinement.\n"
        f"Include: language features, concepts, libraries, frameworks, tools.\n"
        f"Exclude: General computing terms unrelated to Python.\n"
        f"Remove:\n"
        f"- Pure numbers or dates\n"
        f"- Book references or page numbers\n"
        f"- Non-technical and non-Python programming terms\n\n"
        f"Return only the refined list of keywords as list of concepts relevant to Python programming in the following format:\n"
        f"['concept1', 'concept2', 'concept3', ...]\n"  # Example on a single line
    )

    response = client.get_model_response(
        system_prompt="You are a Python expert skilled in identifying and refining programming-related keywords.",
        user_prompt=filter_prompt,
        model_id=model_id,
        max_new_tokens=200,
        temperature=0.5,
        top_k=50,
        top_p=1
    )

    refined_keywords = [kw.strip() for kw in response.strip().split('\n') if kw.strip()]
    return refined_keywords

def generate_and_refine_concepts(keywords: List[str], text: str) -> List[Dict[str, str]]:
    # filter_prompt = (
    #     f"Review these extracted keywords: {keywords}\n"
    #     f"Filter out keywords completely unrelated to Python programming.\n"
    #     f"Polish incomplete or vague keywords to make them precise Python programming concepts.\n"
    #     f"Return a clean, concise list of relevant Python programming concepts."
    # )

    # filtered_keywords_response = client.get_model_response(
    #     system_prompt="You are an expert in Python programming concepts.",
    #     user_prompt=filter_prompt,
    #     model_id=model_id,
    #     max_new_tokens=150,
    #     temperature=0.7,
    #     top_k=50,
    #     top_p=1
    # )
    if isinstance(keywords, str):
        keyword_list = keywords.split('\n')
    else:
        keyword_list = keywords
        
    # Format keywords for prompt
    formatted_keywords = ', '.join(keyword_list)

    additional_concepts_prompt = (
        f"Given the following text chunk and the extracted Python programming keywords:\n\n"
        f"Text Chunk:\n{text}\n\n"
        f"Keywords:\n{keywords}\n\n"
        f"Identify any additional Python programming concepts that are present in the text but not yet captured in the list of keywords.\n"
        f"Focus on unique, meaningful concepts that provide deeper insights into Python programming fundamentals from basic to advanced concepts."
        f"Return the only list of additional concepts relevant to Python programming in the following format:\n"
        f"\n"
        f"['concept1', 'concept2', 'concept3', ...]\n"  # Example on a single line
        )

    additional_concepts_response = client.get_model_response(
        system_prompt="You are an expert at extracting programming concepts.",
        user_prompt=additional_concepts_prompt,
        model_id=model_id,
        max_new_tokens=200,
        temperature=0.5,
        top_k=50,
        top_p=1
    )

    #combined_concepts = set(
    #    keywords.split('\n') + additional_concepts_response.split('\n')
    #)
     
    # Split additional concepts into list
    additional_concepts = [c.strip() for c in additional_concepts_response.split(',')]

    # Combine all concepts
    combined_concepts = keyword_list + additional_concepts
    print('combined concepts',combined_concepts)
    # Deduplicate concepts
    #combined_concepts = deduplicate_concepts(combined_concepts)
    #print('deduplicated concepts',combined_concepts)
    
    detailed_concepts_prompt = (
        f"Given the text chunk, generate detailed descriptions and examples for these Python programming concepts: {list(combined_concepts)}\n\n"
        f"tExtract descriptions and 2-3 examples for each concept directly from the text chunk:\n\n"
        f"Text Chunk:\n{text}\n\n"
        f"Format your response as a JSON-like structure:\n"
        f"{{\n"
        f"  'Concept': 'Concept Name',\n"
        f"  'Description': 'Detailed description from text',\n"
        f"  'Examples': ['Example 1', 'Example 2', 'Example 3']\n"
        f"}}"
    )

    concept_response = client.get_model_response(
        system_prompt="You are an expert at extracting programming concepts, associated descriptionsand examples from a given text chunk.",
        user_prompt=detailed_concepts_prompt,
        model_id=model_id,
        max_new_tokens=2000,
        temperature=0.5,
        top_k=50,
        top_p=1
    )

    return concept_response

# Iterate over the topics and subtopics in the JSON file and extract keywords and concepts
# all_keywords_and_concepts = []
concept_responses = []
#i=0
# Process remaining text chunks
for topic_key, subtopics in tqdm(data.items(), desc="Processing Topics"):
#for topic_key, subtopics in data.items():
        print(f"\n### Processing Topic: {topic_key} ###")
        for subtopic_key, subtopic_value in subtopics.items():
            
            print('topic_key',topic_key)
            #print('subtopic_key',subtopic_key)
            if any(x in topic_key.lower() for x in ['license.txt', 'contents.txt', 'glossary.txt', 'copyright.txt', 'bugs.txt']):
                continue

            chunks = [subtopic_value[i:i+4000] for i in range(0, len(subtopic_value), 4000)]

            #for chunk in chunks:
            for chunk in tqdm(chunks, desc=f"Processing Subtopic: {subtopic_key}", leave=False):
                #i=i+1
                #if i>=5: break
                # Extract keywords
                keywords = extract_keywords(chunk)

        
                #print("Extracted Keywords:", keywords)

                # Filter and refine keywords
                keyword_texts = [kw['text'] for kw in keywords]
                refined_keywords = filter_and_refine_keywords(keyword_texts, subtopic_value)
                #print("Refined Keywords:", refined_keywords)

                # Generate and refine concepts
                concept_response = generate_and_refine_concepts(refined_keywords, subtopic_value)
                #print("Refined Concepts:", concept_response)
                
                # Extract JSON block and parse the response
                json_block = extract_json_block(concept_response)
                if json_block:
                    concept_response = json_block

                parsed_response = parse_response(subtopic_value, chunk, concept_response, topic_key, subtopic_key)
                #print("Parsed Response:", parsed_response)
                concept_responses.extend(parsed_response)
                #print("Concepts Response:", concept_response)

        

# Save the final output to a JSON file
output_file = "keywords_and_concepts.json"
with open(output_file, "w") as f:
    json.dump(concept_responses, f, indent=4)

# with open(output_file, "w") as f:
#     for concept in concepts:
#         # f_out.write(json.dumps({"concept": concept}) + "\n")
#         f_out.write(json.dumps(concept) + "\n")

# f_out.close()
print(f"Results saved to {output_file}")

