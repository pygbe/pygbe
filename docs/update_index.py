import os
import subprocess

#convert README to rst
for file in ['README', 'README_input_format', 'CONTRIBUTING']:
    subprocess.call(['pandoc', '../{}.md'.format(file),
                     '--from', 'markdown', '--to', 'rst',
                     '-s', '-o', '{}.rst'.format(file.lower())])

with open('readme.rst', 'r') as f:
    readme = f.readlines()

readme = readme[readme.index('Installation\n'):]

with open('readme.rst', 'w') as f:
    f.writelines(readme)

os.rename('readme_input_format.rst', 'input_format.rst')
