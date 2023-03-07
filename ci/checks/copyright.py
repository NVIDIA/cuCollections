#!/usr/bin/env python

import sys
import os
import re
from datetime import datetime

# Set the expected ending year to the current year
year = datetime.now().year

# Define the regular expression pattern to match the copyright line
copyright_pattern = re.compile(r"\s*\*\s*Copyright\s*\(c\)\s*(\d{4})(-(\d{4}))?\s*,\s*NVIDIA\s*CORPORATION")

# Loop through all modified files and check the copyright year
for line in sys.stdin:
    # Extract the filename from the input
    file_path = line.strip()

    # Ignore deleted files
    if not os.path.isfile(file_path):
        continue

    # Read the contents of the file
    with open(file_path, 'r') as f:
        content = f.read()

    # Search for the copyright line in the file
    match = copyright_pattern.search(content)

    if match:
        # Extract the starting and ending years from the copyright line
        starting_year = int(match.group(1))
        ending_year = int(match.group(3)) if match.group(3) else starting_year

        # Check if the ending year is up-to-date
        if ending_year != year:
            sys.exit(f'>>>> FAILED: Copyright line in {file_path} is not up-to-date ({ending_year} != {year})')
    else:
        # Copyright line is missing
        sys.exit(f'>>>> FAILED: Copyright line is missing from {file_path}')

print("\n>>>> PASSED: copyright check\n")