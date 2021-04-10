import os
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

import utils

def process(archive):

    name = archive.split(sep='.')[0]

    path = Path(datapath, archive)
    tpath = Path(temp, name)
    with ZipFile(path) as zip:
        zip.extractall(path=tpath)
    
    subpath = archive.split(sep='_')
    outdir = Path(outpath, subpath[0], subpath[1])
    outdir.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(tpath):
        name = file.split(sep='.')[0]
        path = Path(tpath, file)
        with Image.open(path) as img:
            size = img.size
            size = (size[0]//2, size[1]//2)
            img2 = img.resize(size)
            img2.save(Path(outdir, name+'.jpg'), quality=95)
        Path.unlink(path)
    tpath.rmdir()

def main():
    archives = os.listdir(datapath)
    utils.p_umap(process, archives, n_worker=8)


if __name__ == '__main__':

    datapath = 'C:/datasets/eyes'
    temp = 'C:/datasets/temp'
    outpath = 'C:/datasets/eyesout'

    main()