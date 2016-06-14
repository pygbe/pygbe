import os
import sys
import zipfile
import urllib2

def download_zip_with_progress_bar(url):
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()

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
        dl_check = raw_input('The meshes for regression tests don\'t appear to '
                         'be loaded. Would you like to download them from '
                             'Zenodo? (~10MB) (y/n): ')
        if dl_check == 'y':
            mesh_file = 'https://zenodo.org/record/55349/files/pygbe_regresion_test_meshes.zip'
            download_zip_with_progress_bar(mesh_file)
            unzip(mesh_file.split('/')[-1])

            print('Done!')

if __name__ == '__main__':
    check_mesh()
