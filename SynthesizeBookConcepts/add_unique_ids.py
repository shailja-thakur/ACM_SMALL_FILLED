import json
import io

def add_unique_id_optimized(input_file, output_file, buffer_size=10000):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        buffer = io.StringIO()
        counter = 0
        
        for line in infile:
            counter += 1
            
            # Minimal JSON parsing
            try:
                line = line.strip()
                if line.endswith('}'):
                    modified_line = line[:-1] + f', "no": {counter}}}\n'
                else:
                    modified_line = line[:-1] + f', "no": {counter}\n'
                
                buffer.write(modified_line)
                
                # Write buffer to file when it reaches the specified size
                if counter % buffer_size == 0:
                    outfile.write(buffer.getvalue())
                    buffer = io.StringIO()
            
            except Exception as e:
                print(f"Error processing line {counter}: {e}")
        
        # Write any remaining content in the buffer
        outfile.write(buffer.getvalue())

# Usage
input_file = 'synthesized_pythondocs_instruction_responses_testcases-test.json'
output_file = 'synthesized_pythondocs_instruction_responses_testcases-with-ids.json'
add_unique_id_optimized(input_file, output_file)
