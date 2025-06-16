#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
from pathlib import Path
import subprocess

def get_git_root():
    try:
        git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                         stderr=subprocess.DEVNULL,
                                         universal_newlines=True).strip()
        return Path(git_root)
    except subprocess.CalledProcessError:
        return None

def get_git_files(directory):
    try:
        # Get list of tracked files from git
        git_files = subprocess.check_output(['git', 'ls-files'],
                                          cwd=directory,
                                          stderr=subprocess.DEVNULL,
                                          universal_newlines=True)
        return {Path(f.strip()) for f in git_files.splitlines()}
    except subprocess.CalledProcessError:
        return set()

def should_skip_file(file_path, git_files):
    # Skip if file is not tracked by git
    if file_path not in git_files:
        return True

    # Skip patterns matching the workflow's paths-ignore and file exclusions
    skip_patterns = {
        'tsc/meetings/',
        'pendingchanges/',
        '.svg', '.cmd', '.png', '.jpg', '.gif',
        '.mp4', '.pt', '.pth', '.nvdb', '.npz',
        '.wlt'
    }

    str_path = str(file_path)
    return any(pattern in str_path for pattern in skip_patterns)

def fix_whitespace(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip binary-looking files
        if '\0' in content:
            return False

        # Remove trailing whitespace and convert tabs to spaces
        fixed_lines = []
        modified = False

        for line in content.splitlines():
            # Remove trailing whitespace
            new_line = line.rstrip()
            # Convert tabs to spaces (4 spaces per tab)
            new_line = new_line.replace('\t', '    ')

            if new_line != line:
                modified = True
            fixed_lines.append(new_line)

        if modified:
            # Write back only if changes were made
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write('\n'.join(fixed_lines) + '\n')
            print(f"Fixed: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def check_line_length(file_path, max_length=100):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if len(line.rstrip('\n')) > max_length:
                    line_length = len(line.rstrip('\n'))
                    print(f"{file_path}:{line_num}: Line length {line_length} exceeds {max_length} characters")
    except Exception as e:
        print(f"Error checking line length in {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Fix whitespace issues in git-tracked files')
    parser.add_argument('directory', nargs='?', default='.',
                       help='Directory to process (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show files that would be modified without making changes')
    parser.add_argument('--check-length', action='store_true',
                       help='Check for lines exceeding 100 characters')

    args = parser.parse_args()

    # Find git root directory
    git_root = get_git_root()
    if not git_root:
        print("Error: Not a git repository")
        return

    # Convert directory to absolute path
    work_dir = Path(args.directory).resolve()

    # Get list of git-tracked files
    git_files = get_git_files(git_root)
    if not git_files:
        print("Error: No git-tracked files found")
        return

    fixed_count = 0
    processed_count = 0

    for root, _, files in os.walk(work_dir):
        for file in files:
            file_path = Path(root) / file
            try:
                # Convert both paths to absolute before calculating relative path
                relative_path = file_path.resolve().relative_to(git_root)

                if should_skip_file(relative_path, git_files):
                    continue

                processed_count += 1

                if args.check_length:
                    check_line_length(file_path)

                if args.dry_run:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if '\t' in content or any(line.rstrip() != line for line in content.splitlines()):
                            print(f"Would fix: {relative_path}")
                            fixed_count += 1
                    except:
                        continue
                else:
                    if fix_whitespace(file_path):
                        fixed_count += 1
            except ValueError:
                # Skip files that are not under the git root
                continue

    print(f"\nProcessed {processed_count} files")
    if args.dry_run:
        print(f"Would fix {fixed_count} files")
    else:
        print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
