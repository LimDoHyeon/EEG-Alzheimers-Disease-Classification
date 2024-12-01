# 폴더 내 세그먼트 개수 계산기

import os

def count_edf_files(root_folder):
    """
    Count all .edf files in the given folder and its subfolders.

    Parameters:
        root_folder (str): Path to the root directory (e.g., 'AD').

    Returns:
        int: Total count of .edf files.
    """
    edf_count = 0

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.edf'):
                edf_count += 1

    return edf_count

# Example usage
root_folder = "/Users/imdohyeon/Documents/PythonWorkspace/4n/preprocessSeg/NC2"  # Replace with the path to your AD folder
edf_file_count = count_edf_files(root_folder)
print(f"Total .edf files: {edf_file_count}")
