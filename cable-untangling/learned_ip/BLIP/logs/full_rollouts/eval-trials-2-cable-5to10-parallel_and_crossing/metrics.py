import os
import re
import numpy as np

# Function to search and extract the fractions from the log files
def extract_fractions(log_path):
    start_traced_pattern = r"Start correctly traced: (\d+)"
    final_traced_pattern = r"Final correctly traced: (\d+)"
    
    start_traced_list = []
    final_traced_list = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(log_path):
        for file in files:
            if file.endswith('.txt'):  # Assuming log files are .txt files
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Search for the patterns and add to lists
                    start_match = re.search(start_traced_pattern, content)
                    final_match = re.search(final_traced_pattern, content)
                    if start_match:
                        start_traced_list.append(start_match.group(1))
                    if final_match:
                        final_traced_list.append(final_match.group(1))
    return start_traced_list, final_traced_list

# Replace 'your_log_folder_path' with the path to your folder containing the logs
log_folder_path = '.'
start_list, final_list = extract_fractions(log_folder_path)

# Print or process the lists
print("Start correctly traced list:")
print(start_list)
print(np.sum([float(i)/49 for i in start_list])/len(start_list))
print("Final correctly traced list:")
print(final_list)
print(np.sum([float(i)/49 for i in final_list])/len(final_list))
