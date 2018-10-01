import random
import h5py
import numpy as np


def load_data():
    filepath = '../../lung_data.mat'
    f = h5py.File(filepath)

    img_data = f['lung_data']['img'][0]
    imgs = []
    for i in img_data:
        temp = np.array(f[i])
        imgs.append(temp)


    hdata = f['lung_data']['heterogeneityLabel'][0]
    hlabels = []
    for h in hdata:
        hlabels.append(np.array(f[h]))


def main():
    load_data()


if __name__ == '__main__':
    main()
