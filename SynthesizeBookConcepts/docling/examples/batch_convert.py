import json
import logging
import time
from pathlib import Path
from typing import Iterable
import argparse
import re
from docling.datamodel.base_models import ConversionStatus, PipelineOptions
from docling.datamodel.document import ConversionResult, DocumentConversionInput
from docling.document_converter import DocumentConverter

_log = logging.getLogger(__name__)


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            # Export Deep Search document JSON format:
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(conv_res.render_as_dict()))

            # Export Text format:
            with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                fp.write(conv_res.render_as_text())

            # Export Markdown format:
            with (output_dir / f"{doc_filename}.md").open("w") as fp:
                fp.write(conv_res.render_as_markdown())

            # Export Document Tags format:
            with (output_dir / f"{doc_filename}.doctags").open("w") as fp:
                fp.write(conv_res.render_as_doctags())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count


def get_all_files_from_directory(directory, file_extension):
    """
    Recursively finds all files with the given extension in the specified directory and its subdirectories.
    
    :param directory: Path to the directory
    :param file_extension: The file extension to look for (e.g., ".pdf" or ".txt")
    :return: List of Path objects for each file found
    """
    return list(Path(directory).rglob(f"*{file_extension}"))



def chunk_text_based_on_headers(text):
    chunks = {}
    current_chunk = []
    current_header = None

    # Regular expression to match headers and their underlines
    header_patterns = r"^(.*)\n([-=^*~]+)$"

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if the line matches the header pattern
        match = re.match(header_patterns, f"{line}\n{lines[i + 1] if i + 1 < len(lines) else ''}")
        if match:
            # New header found, store the previous chunk
            if current_header:
                chunks[current_header] = "\n".join(current_chunk)

            # Set new header and start a new chunk
            current_header = match.group(1).strip()  # The actual header text
            current_chunk = []

            # Skip the underline line
            i += 1  # Skip next line since it was the underline
        else:
            current_chunk.append(line.strip())
        i += 1

    # Store the last chunk
    if current_header and current_chunk:
        chunks[current_header] = "\n".join(current_chunk)

    return chunks


def read_and_chunk_text_file(file_path):
    """
    Reads the text from a file and chunks it based on header styles.
    
    :param file_path: Path to the text file
    :return: A dictionary with headers as keys and the associated text as values
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return chunk_text_based_on_headers(text)
    # return text



def main():
    logging.basicConfig(level=logging.INFO)

    # Ask user to provide the input directory
    parser = argparse.ArgumentParser(description="Process PDF and text files from a given directory.")
    parser.add_argument("directory", type=str, help="Path to the directory with PDF and text files")
    args = parser.parse_args()

    input_directory = Path(args.directory)

    if not input_directory.is_dir():
        raise FileNotFoundError(f"The provided path '{input_directory}' is not a valid directory.")

    # Get all PDF and text files from the provided directory (including nested directories)
    input_pdf_paths = get_all_files_from_directory(input_directory, ".pdf")
    input_text_paths = get_all_files_from_directory(input_directory, ".txt")

    # Skip specific files by name
    files_to_skip = {"index.txt", "about.txt", "licence.txt"}
    input_text_paths = [path for path in input_text_paths if path.name.lower() not in files_to_skip]

    # Handle PDFs
    if input_pdf_paths:
        doc_converter = DocumentConverter()
        input_pdf = DocumentConversionInput.from_paths(input_pdf_paths)

        start_time = time.time()
        conv_results = doc_converter.convert(input_pdf)
        success_count, partial_success_count, failure_count = export_documents(
            conv_results, output_dir=Path("./scratch")
        )
        end_time = time.time() - start_time
        logging.info(f"All PDFs were converted in {end_time:.2f} seconds.")
        if failure_count > 0:
            raise RuntimeError(
                f"Failed to convert {failure_count} PDFs from {len(input_pdf_paths)}."
            )

    # Handle text files
    if input_text_paths:
        text_chunks = {}
        for text_path in input_text_paths:
            logging.info(f"Processing text file: {text_path}")
            file_chunks = read_and_chunk_text_file(text_path)
            text_chunks[text_path.name] = file_chunks

        # Output the chunked text as JSON
        with open("text_chunks.json", 'w', encoding='utf-8') as json_file:
            json.dump(text_chunks, json_file, indent=4)
        logging.info(f"Text chunks saved as text_chunks.json")

    if not input_pdf_paths and not input_text_paths:
        logging.error("No PDF or text files found to process.")
        return

    

if __name__ == "__main__":
    main()




