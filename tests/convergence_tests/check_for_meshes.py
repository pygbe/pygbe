import os
import sys
import zipfile

try:
    import requests
    import clint
except ImportError:
    sys.exit("Regression tests require `requests` and `clint`, please install using pip or conda")

def download_zip_with_progress_bar(url):
    r = requests.get(url, stream=True)
    path = url.rsplit('/', 1)[-1]
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in clint.textui.progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()

def unzip(meshzip):
    with zipfile.ZipFile(meshzip, 'r') as myzip:
        myzip.extractall(path='./tmp')

    print('Unzipping meshes to folder \'geometry/\'...')
    os.rename(os.path.join('tmp', 'regresion_tests_meshes'), 'geometry')
    os.rmdir('tmp')
    print('Removing zip file...')
    os.remove(meshzip)

def check_mesh():
    #check if there's a geometry folder present in the directory
    if not os.path.isdir('geometry'):
        dl_check = input('The meshes for regression tests don\'t appear to '
                         'be loaded. Would you like to download them from '
                             'Zenodo? (~10MB) (y/n): ')
        if dl_check == 'y':
            mesh_file = 'https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip'
            download_zip_with_progress_bar(mesh_file)
            unzip(mesh_file.split('/')[-1])

            print('Done!')

if __name__ == '__main__':
    check_mesh()
