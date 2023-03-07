#!/usr/bin/env python

# Copyright (c) 2023, NVIDIA CORPORATION.

import sys
import os
import re
from datetime import datetime

# Set the expected ending year to the current year
year = datetime.now().year

# Define the regular expression pattern to match the copyright line
copyright_pattern = re.compile(r"\s*(\*|#)\s*Copyright\s*\(c\)\s*(\d{4})(-(\d{4}))?\s*,\s*NVIDIA\s*CORPORATION")

exit_code = 0
# Loop through all modified files and check the copyright year
for file_path in sys.argv[1:]:
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
        starting_year = int(match.group(2))
        ending_year = int(match.group(4)) if match.group(4) else starting_year

        # Check if the ending year is up-to-date
        if ending_year != year:
            print(f'warning: Copyright line in {file_path} is not up-to-date ({ending_year} != {year})')
            exit_code = 1
    else:
        # Copyright line is missing
        print(f'warning: Copyright line is missing from {file_path}')
        exit_code = 1

sys.exit(exit_code)