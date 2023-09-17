import os
import zipfile
import shutil
from PIL import Image

def remove_empty_files(rootdir):
    count = 0
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            fullname = os.path.join(root, f)
            try:
                #print(fullname)
                img = Image.open(fullname)
                img.verify()     # to veify if its an img
                img.close()     #to close img and free memory space
            except (IOError, SyntaxError) as e:
                print('Bad file:', fullname)
                count += 1
                os.remove(fullname)
    print('Number of files of 0 bytes: ', count)

if __name__ == "__main__":

    root_dir = os.path.join(os.getcwd(), '')
    # Create directory for dataset
    data_dir = os.path.join(root_dir, 'src/datasets')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Extract content of zip file
    path_to_zip_file = os.path.join(root_dir, 'data/notMNIST.zip')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    shutil.rmtree(os.path.join(data_dir, '__MACOSX')) # remove MACOSX directory
    # Rename train and test data directories to lower-case letters
    os.rename(os.path.join(data_dir, 'notMNIST/Train'), os.path.join(data_dir, 'notMNIST/train'))
    os.rename(os.path.join(data_dir, 'notMNIST/Test'), os.path.join(data_dir, 'notMNIST/test'))

    # Filter images that are empty
    remove_empty_files(os.path.join(data_dir, 'notMNIST'))