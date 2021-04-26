from bs4 import BeautifulSoup
import requests
import re

from multiprocessing import Pool, cpu_count
import gc
from tqdm.auto import tqdm
import os

def p_umap(func, data, n_worker=1, chunk=50):
    with Pool(n_worker) as pool:
        out = list(tqdm(pool.imap_unordered(func, data, chunksize=chunk), total=len(data)))
    return out

inpath = '/media/ermanno/64c076a6-5264-4d0b-a7eb-43621a3dd20e/lavidb'
outpath = '/home/ermanno/Desktop/datasets/eyes2'
inputfiles = [inpath + '/' + file for file in sorted(os.listdir(inpath))]
n_samples = 10

from PIL import Image
import torchvision.transforms.functional as TF
def load_sequence(idx, fz, basename, start, end):
    dirname = str(idx).zfill(4)
    outdir = os.path.join(outpath, dirname)
    os.mkdir(outdir)

    files = [fz.open(basename+'_'+str(i).zfill(3)+'.tif') for i in range(start, end+1)]
    
    imgs = [Image.open(fp) for fp in files]

    size = imgs[0].size
    size = size[1], size[0]

    TF.resize(imgs[3], size=(size[0]//2, size[1]//2)).save(os.path.join(outdir, 'label.png'))

    [TF.resize(imgs[i], size=(size[0]//4, size[1]//4)).save(os.path.join(outdir, 'im{}_x{}.png'.format(i+1, 2))) for i in range(7)]
    [TF.resize(imgs[i], size=(size[0]//8, size[1]//8)).save(os.path.join(outdir, 'im{}_x{}.png'.format(i+1, 4))) for i in range(7)]

    # print(start, end)
    for file in files:
        file.close()


import zipfile
import random
def load_zip(idx):
    print(inputfiles[idx])
    zipname = inputfiles[idx]
    basename = os.path.basename(zipname)
    parts = basename.split('_')
    basename = parts[0] + '_' + parts[1]

    fz = zipfile.ZipFile(zipname)
    used = set()
    for i in range(n_samples):
        while True:
            start = random.randint(0, 1000-7)
            end = start + 6
            if not (start in used or end in used):
                load_sequence(idx*n_samples + i, fz, basename, start, end)
                for i in range(start, end+1): 
                    used.add(i)
                break
    
    fz.close()



if __name__=='__main__':

    p_umap(load_zip, list(range(len(inputfiles))), n_worker=4, chunk=1)