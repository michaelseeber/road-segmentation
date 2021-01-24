from os.path import splitext
from os import listdir
from os.path import join
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
# from PIL import Image

from numpy import load
import torch as th

from PIL import Image
from numpy import array

class NPYDataset(Dataset):
    def __init__(self,
                 dir_npy_x, dir_npy_y=None,
                 inds_x=None, inds_y=None,
                 func_x=None, func_y=None,
                 model_gen=None, disc_output_shape=tuple([1])):

        self.dir_npy_x = dir_npy_x
        self.dir_npy_y = dir_npy_y
        self.inds_x, self.inds_y = inds_x, inds_y
        self.func_x, self.func_y = func_x, func_y
        self.ids = [splitext(file)[0] for file in listdir(dir_npy_x)
                    if not file.startswith('.')]
        self.model_gen = model_gen
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.disc_output_shape = disc_output_shape

    def __len__(self):
        return len(self.ids)

    # @classmethod
    def preprocess(self, px, func=None):

        px_t = px.transpose((2, 0, 1))
        if func is not None:
            # print('before: ', px_t.shape)
            px_t = func(px_t)
            # print('after :', px_t.shape)

        return px_t.astype(np.float64)

    def __getitem__(self, i):
        idx = self.ids[i]
        # print('...call made...:', i)

        file_x = join(self.dir_npy_x, idx + '.npy')

        file_y = file_x if self.dir_npy_y is None else join(self.dir_npy_y, idx + '.npy')
        # assert len(file_x) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {file_x}'
        npy_x = load(file_x)
        x = npy_x if self.inds_x is None else npy_x[:, :, self.inds_x]
        x = self.preprocess(x, self.func_x)
        x_as_t = th.from_numpy(x)

        npy_y = npy_x if file_x == file_y else load(file_y)
        y = npy_y if self.inds_y is None else npy_y[:, :, self.inds_y]
        y = self.preprocess(y, self.func_y)
        y_as_t = th.from_numpy(y)

        assert x.shape[-2:] == y.shape[-2:], \
            f'Image and mask {idx} should be the same size, but are {x.size} and {y.size}'


        return {'x': x_as_t, 'y': y_as_t}

class PNGDataset(Dataset):
    def __init__(self,
                 dir_npy_x, dir_npy_y=None,
                 func_x=None, func_y=None,
                 model_gen=None, disc_output_shape=tuple([1])):

        self.dir_npy_x = dir_npy_x
        self.dir_npy_y = dir_npy_y
        self.func_x, self.func_y = func_x, func_y
        self.ids = [splitext(file)[0] for file in listdir(dir_npy_x)
                    if not file.startswith('.') and file.endswith('.png')]
        self.model_gen = model_gen
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.disc_output_shape = disc_output_shape

    def __len__(self):
        return len(self.ids)

    # @classmethod
    def process(self, px, func=None):

        px_ = array(px)

        px_t = px_.transpose((2, 0, 1)) if len(px_.shape) > 2 else px_
            
        if func is not None:
            px_t = func(px_t)

        if px_t.max() > 1:
            px_t = px_t / 255

        return px_t.astype(np.float64)

    def __getitem__(self, i):
        idx = self.ids[i]
        # print('...call made...:', i)

        file_x = join(self.dir_npy_x, idx + '.png')
        file_y = None if self.dir_npy_y is None else join(self.dir_npy_y, idx + '.png')

        png_x = Image.open(file_x)
        if file_y is not None:
            png_y = Image.open(file_y)

        x = self.process(png_x, self.func_x)
        x_as_t = th.from_numpy(x)

        if file_y is not None:
            y = self.process(png_y, self.func_y)
            y_as_t = th.from_numpy(y)

        # assert x.shape[-2:] == y.shape[-2:], \
        #     f'Image and mask {idx} should be the same size, but are {x.size} and {y.size}'

        if file_y is not None:
            return {'x': x_as_t, 'y': y_as_t}
        else:
            return {'x': x_as_t}

# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt

    DIR_X = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/training/images_augmented'
    DIR_Y = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/training/groundtruth_augmented'

    # file = '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/training/images_augmented/satImageAugmented_1000.png'
    # img = array(Image.open(file))


    dataset = PNGDataset(DIR_X) #, DIR_Y)
    n_dat = len(dataset)
    n_train = int(0.8 * n_dat)
    batch_size = 2

    train, val = random_split(dataset, [n_train, n_dat - n_train])
    train_loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    
    print(n_train)
    for i, batch in enumerate(train_loader):
        x = batch['x']
        # y = batch['y']
        plt.imshow(x[0].numpy().transpose(1, 2, 0))
        plt.show()
        # plt.imshow(y[0].numpy())
        # plt.show()
        # print(x.shape) 
        # print(x)
        # print(y.shape)
        if i==0:
            break
