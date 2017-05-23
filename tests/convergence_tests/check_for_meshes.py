import os
import sys
import zipfile

try:
    import requests
    import clint
except ImportError:
    sys.exit("Convergence tests require `requests` and `clint`, please install using pip or conda")

def download_zip_with_progress_bar(url):
    r = requests.get(url, stream=True)
    path = url.rsplit('/', 1)[-1]
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in clint.textui.progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()

def unzip(meshzip, folder_name, rename_folder):
    '''Unzip a zip file

    Arguments:
    ----------
    meshzip      : file we want to unzip
    folder_name  : str, name of the folder after extracting.
                  We want this to modify it when download, to follow pygbe 
                  problem folder design for running.
    rename_folder: str, name of the folder were we want to store the download.
                   We rename the folder_name, to match config files. 
    '''
    with zipfile.ZipFile(meshzip, 'r') as myzip:
        myzip.extractall(path='./tmp')

    print('Unzipping meshes to folder \'geometry/\'...')
    os.rename(os.path.join('tmp', folder_name), rename_folder)
    os.rmdir('tmp')
    print('Removing zip file...')
    os.remove(meshzip)

def check_mesh(mesh_file, folder_name, rename_folder, size):
    ''' Check if there's a geometry folder present in the directory.

    Arguments:
    ----------
    mesh_file    : str, mesh_file url
    folder_name  : str, name of the folder after extracting.
                   We want to pass this to be used in unzip function, where 
                   we rename it.
    rename_folder: str, name of the folder were we want to store the download.
                   We rename the folder_name, to match config files. 
    size         : str, estimate size of file    
    '''
    if not os.path.isdir(rename_folder):
        dl_check = input('The meshes for convergence tests don\'t appear to '
                         'be loaded. Would you like to download them from '
                             'Zenodo?(' + size + ')(y/n): ')
        if dl_check == 'y':
            download_zip_with_progress_bar(mesh_file)
            unzip(mesh_file.split('/')[-1], folder_name, rename_folder)

            print('Done!')
