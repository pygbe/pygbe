import os
import subprocess

#convert README to rst
for file in ['README', 'README_input_format', 'CONTRIBUTING']:
    subprocess.call(['pandoc', '../{}.md'.format(file),
                     '--from', 'markdown', '--to', 'rst',
                     '-s', '-o', '{}.rst'.format(file.lower())])

os.rename('README_input_format.rst', 'input_format.rst')

with open('README.rst', 'r') as f:
    readme = f.read()

readme = readme.split('Installation', 1)

with open('index_intro.rst', 'w') as f:
    f.write(readme[0])

with open('index_outro.rst', 'w') as f:
    f.write(readme[1])
