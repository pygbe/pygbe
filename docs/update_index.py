import os
import subprocess

#remove DOI badge from readme before conversion to RST
with open('../README.md', 'r') as f:
    readme = f.readlines()

for line in readme:
    if '[DOI]' in line:
        readme.remove(line)

with open('../README.md', 'w') as f:
    f.writelines(readme)

#convert README to rst
for file in ['README', 'README_input_format', 'CONTRIBUTING']:
    subprocess.call(['pandoc', '../{}.md'.format(file),
                     '--from', 'markdown', '--to', 'rst',
                     '-s', '-o', '{}.rst'.format(file.lower())])

os.rename('readme_input_format.rst', 'input_format.rst')

with open('readme.rst', 'r') as f:
    readme = f.read()

readme = readme.split('Installation', 1)

intro, installation = readme

installation, references = installation.split('References', 1)

with open('intro.rst', 'w') as f:
    f.write(intro)

with open('installation.rst', 'w') as f:
    f.write('Installation')
    f.write(installation)

with open('references.rst', 'w') as f:
    f.write('References')
    f.write(references)

os.remove('readme.rst')
