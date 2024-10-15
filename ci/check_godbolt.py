#!/usr/bin/env python

import os
import json
import requests
import git
import re
import argparse

def open_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def write_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write(data)

def create_compiler_explorer_shortlink(file_path, COMPILER_ID, COMPILER_FLAGS):
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
                        "id": COMPILER_ID,
                        "options": COMPILER_FLAGS,
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

def update_readme_with_shortlinks(readme_path, shortlink_table):
    # Open and read the README.md file
    readme_content = open_file(readme_path)
    lines = readme_content.splitlines()

    updated_lines = []
    # Iterate through each line of the README
    for line in lines:
        updated_line = line
        # Check each file in the shortlink_table
        for file_name, shortlink in shortlink_table.items():
            # Check if the file name exists in the line
            if file_name in line:
                # Look for the specific godbolt link format and replace the URL
                updated_line = re.sub(
                    r"\(see \[live example in godbolt\]\(https?://[^\)]+\)\)",
                    f"(see [live example in godbolt]({shortlink}))",
                    line
                )
        updated_lines.append(updated_line)

    # Join the lines and write the updated content back to README.md
    updated_content = "\n".join(updated_lines)
    write_file(readme_path, updated_content)

COMPILER_ID = 'nvcc125u1'  # Desired compiler ID
COMPILER_FLAGS = '-std=c++17 -arch=sm_70 --expt-extended-lambda'  # NVCC compiler flags
EXAMPLES_DIR = "examples"  # Path to the examples directory relative to the repo root

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate shortlinks for example CUDA files.")
    parser.add_argument('--force', action='store_true', help="Generate new links for all example files, even if unchanged.")
    args = parser.parse_args()

    # Initialize the Git repository
    repo = git.Repo(search_parent_directories=True)
    repo_root = repo.git.rev_parse("--show-toplevel")
    
    # Ensure we are in the ci directory
    ci_dir = os.path.join(repo_root, 'ci')
    os.chdir(ci_dir)

    # Check if the remote repository already exists
    try:
        remote = repo.remote("gold_cuco")
    except git.exc.NoSuchRemoteError:
        # Create the remote repository if it doesn't exist
        remote = repo.create_remote("gold_cuco", "https://github.com/NVIDIA/cuCollections.git")
        print("Remote 'gold_cuco' created.")

    # Fetch the remote repository
    print("Fetching latest changes from the remote 'gold_cuco'...")
    remote.fetch()

    # Get the current branch
    current_branch = repo.active_branch

    # Get the 'dev' branch from the remote
    dev_branch = remote.refs.dev

    # Initialize a hash table (dictionary) to store file shortlinks
    shortlink_table = {}

    # Determine which files to process based on the --force flag
    if args.force:
        # Get all .cu and .cuh files in the examples directory
        example_files = []
        for root, _, files in os.walk(os.path.join(repo_root, EXAMPLES_DIR)):
            for file_name in files:
                if file_name.endswith('.cu'):
                    example_files.append(os.path.relpath(os.path.join(root, file_name), repo_root))
    else:
        # Get the list of changed files between the current branch and the 'dev' branch
        changed_files = current_branch.commit.diff(dev_branch.commit)
        example_files = [diff.b_path for diff in changed_files if diff.b_path.startswith(EXAMPLES_DIR) and (diff.b_path.endswith('.cu'))]

    # Iterate through the example files and create short links
    for file_path in example_files:
        full_file_path = os.path.join(repo_root, file_path)
        shortlink = create_compiler_explorer_shortlink(full_file_path, COMPILER_ID, COMPILER_FLAGS)
        if shortlink:
            shortlink_table[file_path] = shortlink  # Store the file and its shortlink in the hash table
            print(f"File: {file_path}: Shortlink: {shortlink}")

    # Update README.md with the shortlinks
    readme_path = os.path.join(repo_root, "README.md")
    update_readme_with_shortlinks(readme_path, shortlink_table)

if __name__ == "__main__":
    main()

