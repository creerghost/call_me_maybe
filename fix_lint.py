import os
import textwrap

def fix_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if len(line.rstrip('\n')) > 79:
            # If it's a docstring line, we can wrap it
            if line.strip().startswith(('Args:', 'Returns:', 'Raises:', '"""""', "'''")):
                new_lines.append(line)
            elif '"""' in line or "'''" in line or line.strip().startswith('*') or line.strip().startswith('-') or line.strip().startswith(('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')):
                # indent level
                indent = len(line) - len(line.lstrip())
                wrapped = textwrap.fill(line.strip(), width=79 - indent, subsequent_indent=' ' * indent)
                # re-add original indent to first line
                new_lines.append((' ' * indent) + wrapped + '\n')
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

import glob
for f in glob.glob("src/**/*.py", recursive=True):
    fix_file(f)
