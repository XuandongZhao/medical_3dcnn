import h5py
import numpy as np


def load_data():
    filepath = '../../lung_data.mat'
    f = h5py.File(filepath)

    img_data = f['lung_data']['image'][0]
    imgs = []
    max_z = 0
    max_w = 0
    max_h = 0
    for i in img_data:
        temp = np.array(f[i], dtype=np.float32)
        imgs.append(temp)
        if temp.shape[0] > max_z:
            max_z = temp.shape[0]
        if temp.shape[1] > max_w:
            max_w = temp.shape[1]
        if temp.shape[2] > max_h:
            max_h = temp.shape[2]
    for i in range(len(imgs)):
        padz0 = int((max_z - imgs[i].shape[0]) / 2)
        padz1 = max_z - imgs[i].shape[0] - padz0
        padw0 = int((max_w - imgs[i].shape[1]) / 2)
        padw1 = max_w - imgs[i].shape[1] - padw0
        padh0 = int((max_h - imgs[i].shape[2]) / 2)
        padh1 = max_h - imgs[i].shape[2] - padh0
        npad = ((padz0, padz1), (padw0, padw1), (padh0, padh1))
        imgs[i] = np.pad(imgs[i], pad_width=npad, mode='constant', constant_values=0)
    imgs = np.asarray(imgs)

    labeldata = f['lung_data']['survivalLabel'][0]
    labels = []
    for l in labeldata:
        labels.append(np.array(f[l], dtype=np.int8))
    labels = np.asarray(labels).reshape(-1)
    print('images shape: ', imgs.shape)
    print('labels shape: ', labels.shape)
    return imgs, labels, max_z, max_w, max_h


def main():
    load_data()


if __name__ == '__main__':
    main()

# visualize
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# def make_ax(grid=False):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.grid(grid)
#     return ax
# ax = make_ax(True)
# ax.voxels(imgs[61], edgecolors='gray')
# plt.show()
