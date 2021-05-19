#!/usr/bin/env python3
import os
from markdown_toclify import markdown_toclify

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = root_dir + "/include/tim/vx/ops"
template_file = "/docs/Operators.md.template"
md_file = "/docs/Operators.md"

def iterfindfiles(path, fnexp):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(fnexp):
                yield filename


def find_files_by_pattern(pattern, path='.'):
    paths = []
    for filename in iterfindfiles(path, pattern):
        paths.append(filename)
    paths.sort()
    return paths

BLOCK_NONE = 0
BLOCK_START = 1
def get_md_blocks(file):
    blocks = []
    with open(file) as fhndl:
        lines = fhndl.readlines()
        lines_out = []
        status = BLOCK_NONE
        for index, line in enumerate(lines):
            if (status == BLOCK_NONE):
                if (line.strip() == "/**"):
                    status = BLOCK_START
            else: # status == BLOCK_START
                if (line.strip() == "*/"):
                    status = BLOCK_NONE
                    blocks.append(lines_out.copy())
                    lines_out = []
                elif (line.lstrip().startswith("*")):
                    if (line.lstrip().startswith("*\n")):
                        lines_out.append("\n")
                    else:
                        lines_out.append(line.lstrip()[2:])
                else:
                    status = BLOCK_NONE
                    lines_out = []
    return blocks

all_blocks = []
for path in find_files_by_pattern(('.cpp', '.h'), path=src_dir):
    blocks = get_md_blocks(src_dir + "/" + path)
    all_blocks.extend(blocks)

with open(root_dir + template_file) as fhndl:
    lines = fhndl.readlines()

new_lines = lines.copy()
offset = 0
for index, line in enumerate(lines):
    if line.find("{DOCS}") != -1:
        # new_lines[index] = new_lines[index].replace('{DOCS}', '{DOCS1}')
        del new_lines[index + offset]
        for blocks in all_blocks:
            for line in blocks:
                new_lines.insert(index + offset, line)
                offset += 1
            new_lines.insert(index + offset, '\n')
            offset += 1
        offset -= 1

with open(root_dir + md_file, mode='w',newline='\n', encoding='UTF-8') as fhndl:
    fhndl.writelines(new_lines)

cont = markdown_toclify(input_file=root_dir + md_file)

with open(root_dir + md_file, mode='w',newline='\n', encoding='UTF-8') as fhndl:
    fhndl.write(cont)
