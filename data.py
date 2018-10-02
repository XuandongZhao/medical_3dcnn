import h5py
import numpy as np
import random


def load_data(train_num, label_name):
    filepath = '../lung_data.mat'
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

    labeldata = f['lung_data'][label_name][0]  # ajccLabelSim survivalLabel
    labels = []
    for l in labeldata:
        labels.append(np.array(f[l], dtype=np.int8))
    labels = np.asarray(labels).reshape(-1)
    print('images shape: ', imgs.shape)
    print('labels shape: ', labels.shape)
    idx = np.random.permutation(len(labels))
    train_idx = idx[0:train_num]
    test_idx = idx[train_num:]
    train_imgs = [imgs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_imgs = [imgs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_imgs, train_labels, test_imgs, test_labels, max_z, max_w, max_h


def load_balance_data(train_num, label_name, class_num=2):
    train_imgs, train_labels, test_imgs, test_labels, max_z, max_w, max_h = load_data(train_num, label_name)
    train_data = [[] for i in range(class_num)]
    for i in range(len(train_imgs)):
        tempdata = {}
        tempdata['img'] = train_imgs[i]
        tempdata['label'] = train_labels[i]
        if train_labels[i] == 2:
            continue
        train_data[train_labels[i]].append(tempdata)

    maxtrainlen = 0
    maxtrainidx = 0
    for i in range(len(train_data)):
        if len(train_data[i]) > maxtrainlen:
            maxtrainlen = len(train_data[i])
            maxtrainidx = i
    all_train_data = []
    for i in range(maxtrainlen):
        for j in range(len(train_data)):
            if j == maxtrainidx:
                all_train_data.append(train_data[j][i])
            else:
                tempidx = random.randint(0, len(train_data[j]) - 1)
                all_train_data.append(train_data[j][tempidx])
    print('all train data %d' % len(all_train_data))

    test_data = [[] for i in range(class_num)]
    for i in range(len(test_imgs)):
        tempdata = {}
        tempdata['img'] = test_imgs[i]
        tempdata['label'] = test_labels[i]
        if test_labels[i] == 2:
            continue
        test_data[test_labels[i]].append(tempdata)

    maxtestlen = 0
    maxtestidx = 0
    for i in range(len(test_data)):
        if len(test_data[i]) > maxtestlen:
            maxtestlen = len(test_data[i])
            maxtestidx = i
    all_test_data = []
    for i in range(maxtestlen):
        for j in range(len(test_data)):
            if j == maxtestidx:
                all_test_data.append(test_data[j][i])
            else:
                tempidx = random.randint(0, len(test_data[j]) - 1)
                all_test_data.append(test_data[j][tempidx])
    print('all test data %d' % len(all_test_data))

    return all_train_data, all_test_data, max_z, max_w, max_h


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
