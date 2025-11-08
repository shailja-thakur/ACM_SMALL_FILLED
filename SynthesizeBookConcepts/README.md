# Curriculum Concept Generator

A tool for automatically generating structured curriculum concepts from various educational sources including books, specifications, and tutorials. The tool processes both structured (text, markdown, tabular) and unstructured (PDF, HTML, OCR) inputs to extract meaningful educational concepts with examples and descriptions.

## 🌟 Features

- Extract concepts from various input formats (currently supporting PDF and text files)
- Generate structured concept representations with descriptions and examples
- Synthesize problems, solution code, and test cases from concepts
- Support for batch processing and cloud deployment
- Intelligent keyword extraction and concept refinement

## 📁 Project Structure

```
.
├── docling/                      # Documentation and book parsing utilities
├── __pycache__/                 # Python cache directory
├── add_unique_ids.py            # Utility for adding IDs to concepts
├── extract_keywords_updated.py   # Main concept extraction script
├── extract_keywords_gen_tree.py  # Tree-based keyword extraction
├── ossinstruct_v6.py            # Problem synthesis script
├── utils.py                     # Common utilities
├── __init__.py                  # Package initialization
├── script_add_ids.sh            # Script for adding unique IDs
├── generate_concepts.sh         # Concept generation job script
├── synthesize_code.sh          # Code synthesis job script
└── text_chunks_v1.json         # Input text chunks
```

## 🚀 Getting Started

### Concept Extraction

To extract concepts from your input files:

```bash
python extract_keywords_updated.py
```

This will process your input and generate a structured JSON output with concepts, descriptions, and examples.

Example output structure:
```json
{
  "Concept": "Binary Floating Point",
  "Description": "Binary floating-point numbers are represented in computer hardware as base 2 (binary) fractions. For example, the decimal fraction '0.625' can be represented as the binary fraction '0.101'.",
  "Examples": [
    "0.625 = 6/10 + 2/100 + 5/1000",
    "0.101 = 1/2 + 0/4 + 1/8"
  ]
}
```

### Problem Synthesis

To generate problems, solution code, and test cases from concepts:

```bash
python ossinstruct_v6.py \
    --instruct_mode "C->I" \
    --seed_data_file text_chunks_v1.json \
    --tag concept_gen \
    --temperature 0.7 \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --num_batched_requests 32 \
    --seed_code_start_index 0 \
    --num_sample_per_request 1 \
    --model_type vllm
```

## ☁️ Cloud Deployment

For running jobs on cloud infrastructure:

1. Concept Generation:
```bash
./generate_concepts.sh
```

2. Code Synthesis:
```bash
./synthesize_code.sh
```

## 🔧 Dependencies

- Python 3.x
- spaCy
- PyTextRank
- tqdm
- VLLM
- Additional requirements listed in requirements.txt (to be created)

## 📝 Note

- Currently supports PDF and text file inputs
- Optimized for programming curriculum generation (e.g., Python concepts)
- Supports batch processing for large-scale concept generation
- Includes utilities for ID management and concept deduplication

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

[License Information to be added]
