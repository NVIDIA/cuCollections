#!/usr/bin/env python

import os
import json
import requests
import subprocess

def open_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def create_compiler_explorer_shortlink(file_path, compiler_id, compiler_flags):
    source_code = open_file(file_path)
    # Step 1: Prepare the JSON payload
    payload = {
        "sessions": [
            {
                "id": 1,
                "language": "cuda",
                "source": source_code,
                "compilers": [
                    {
                        "id": compiler_id,
                        "options": compiler_flags,
                        "libs": [
                            {
                                "id": "cccl",
                                "version": "trunk"
                            },
                            {
                                "id": "cuco",
                                "version": "dev"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # Step 2: Send a POST request to the /api/shortener endpoint
    url = 'https://godbolt.org/api/shortener'
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Step 3: Parse the response
    if response.status_code == 200:
        data = response.json()
        short_url = data.get('url')
        if short_url:
            return short_url
        else:
            print("Error: Short URL not found in response.")
            return None
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return None

compiler_id = 'nvcc125u1'  # Replace with the desired compiler ID
compiler_flags = '-std=c++17 -arch=sm_70 --expt-extended-lambda'  # Replace with your compiler flags
EXAMPLES_DIR = "examples"

def get_changed_cuda_files():
    # Run the git command to get the changed files
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
        capture_output=True,
        text=True
    )

    # Check if the command was successful
    if result.returncode != 0:
        print("Error executing git command:", result.stderr)
        return []

    # Get the output and filter for .cu files in EXAMPLES_DIR
    return result.stdout.splitlines()

def main():
    changed_files = get_changed_cuda_files()

    for file in changed_files:
        shortlink = create_compiler_explorer_shortlink(file, compiler_id, compiler_flags)
        print(f"File: {file}: Shortlink: {shortlink}")

if __name__ == "__main__":
    main()
