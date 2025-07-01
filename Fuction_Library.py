import shutil
from scipy.ndimage import generic_filter
import torch
import os
import numpy as np
import random
from sklearn import metrics
import scipy.io as sio
from skimage import io
import cv2
from scipy.ndimage import gaussian_filter


def get_data(DATA_PATH):
    setup_seed(282)
    dataset = os.path.splitext(os.path.basename(DATA_PATH))[0]
    if dataset == 'Shuguang':
        mat = sio.loadmat(DATA_PATH+'/data.mat')
        t1 = np.array(mat['im1'], dtype=np.float32)[:, :, 0]
        t2 = np.array(mat['im2'], dtype=np.float32)
        ref = np.array(mat['im3'], dtype=float) / 255.0
        t1 = (t1 / 255.0) * 2.0 - 1.0
        t1 = t1[:, :, np.newaxis]
        t2 = (t2 / 255.0) * 2.0 - 1.0

    elif dataset == 'France':
        mat = sio.loadmat(DATA_PATH + '/data.mat')
        t1 = np.array(mat['im1'], dtype=np.float32)
        t2 = np.array(mat['im2'], dtype=np.float32)
        ref = np.array(mat['im3'], dtype=float)
        t1 = (t1 / 255.0) * 2.0 - 1.0
        t2 = (t2 / 255.0) * 2.0 - 1.0

    elif dataset == 'Texas':
        mat = sio.loadmat(DATA_PATH+'/data.mat')
        ref = np.array(mat["ROI_1"], dtype=float)
        t1 = np.array(mat["t1_L5"], dtype=np.float32)
        t2 = np.array(mat["t2_ALI"], dtype=np.float32)
        nc1 = t1.shape[2]
        nc2 = t2.shape[2]
        temp1 = np.reshape(t1, (-1, nc1))
        temp2 = np.reshape(t2, (-1, nc2))
        limits = np.mean(temp1, 0) + 3.0 * np.std(temp1, 0)
        for channel, limit in enumerate(limits):
            temp = temp1[:, channel]
            temp[temp > limit] = limit
            temp = 2.0 * temp / np.max(temp) - 1.0
            temp1[:, channel] = temp
        limits = np.mean(temp2, 0) + 3.0 * np.std(temp2, 0)
        for channel, limit in enumerate(limits):
            temp = temp2[:, channel]
            temp[temp > limit] = limit
            temp = 2.0 * temp / np.max(temp) - 1.0
            temp2[:, channel] = temp
        t1 = np.reshape(temp1, np.shape(t1))
        t2 = np.reshape(temp2, np.shape(t2))
        del temp1, temp2, limits, temp

    elif dataset == 'Guizhou':
        mat = sio.loadmat(DATA_PATH + '/data.mat')
        t1 = np.array(mat["im1"], dtype=np.float32)
        t2 = np.array(mat["im2"], dtype=np.float32)
        ref = np.array(mat["im3"], dtype=float)
        t1 = (t1 / 255.0) * 2.0 - 1.0
        t2 = (t2 / 255.0) * 2.0 - 1.0

        t1 = t1[:, :, np.newaxis]
        t2 = t2[:, :, np.newaxis]

    elif dataset == 'Mexico':
        t1 = cv2.imread('Data/Mexico/im1.bmp')[:, :, 0]
        t2 = cv2.imread('Data/Mexico/im2.bmp')[:, :, 0]
        ref = cv2.imread('Data/Mexico/im3.bmp', cv2.IMREAD_GRAYSCALE) / 255.0

        t1 = (t1 / 255.0) * 2.0 - 1.0
        t2 = (t2 / 255.0) * 2.0 - 1.0

        t1 = t1[:, :, np.newaxis]
        t2 = t2[:, :, np.newaxis]

    else:
        print("==> For a new dataset, we recommend a similar pre-processing as Shuguang dataset!")
        return 0

    return t1, t2, ref


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalization(data):
    normalised_data = (data - data.min()) / (data.max() - data.min())
    return normalised_data


def create_folder(path):
    full_path = os.path.join(path, 'exp')
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    os.makedirs(full_path)
    return full_path


def change_map(difference_img):
    _, map2 = cv2.threshold(
        (difference_img * 255).type(torch.uint8).numpy(),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    map2 = map2 / 255.0
    return map2


def calculate_metric(ref_map, change_map, name):

    ref_label, map_label = ref_map.flatten(), change_map.flatten()
    confusion_matrix = metrics.confusion_matrix(ref_label, map_label)

    tmp_FP = confusion_matrix[0,1]
    tmp_FN = confusion_matrix[1,0]
    tmp_OE = len(ref_label) - np.trace(confusion_matrix)
    tmp_OA = np.trace(confusion_matrix) / len(ref_label)
    tmp_KC = metrics.cohen_kappa_score(ref_label, map_label)
    tmp_F1 = metrics.f1_score(ref_label, map_label)

    message = ""
    message += f"{name} results ==>\n"

    message += "FP is : {:d}\n".format(tmp_FP)
    message += "FN is : {:d}\n".format(tmp_FN)
    message += "OE is : {:d}\n".format(tmp_OE)
    message += "OA is : {:.4f}\n".format(tmp_OA)
    message += "F1 is : {:.4f}\n".format(tmp_F1)
    message += "KC is : {:.4f}\n".format(tmp_KC)

    print(message)

    return tmp_KC, message


def write_images(img, SAVE_PATH, img_name, img_type):
    if img_type == 'img':
        img = (0.5 * (img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() + 1) * 255).astype(np.uint8)
        if img.shape[2] > 3:
            img = img[:, :, 1:4]
            # img = img[:, :, [3, 4, 7]]  # 3,4,7 for Texas
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
    elif img_type == 'map':
        img = (img * 255).astype(np.uint8)
    io.imsave(SAVE_PATH + f'/{img_name}.png', img)


def cam(x):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


def combine_maps(cm1, cm2):

    map = np.zeros_like(cm1, dtype=np.uint8)

    map[(cm1 == 1) & (cm2 == 1)] = 1
    map[(cm1 == 0) & (cm2 == 0)] = 0
    map[(cm1 != cm2)] = 2

    return map


def classify_patch(patch):

    alpha = 0.8
    total_count = len(patch)
    center_value = patch[total_count // 2]
    count_center = np.sum(patch == center_value)

    if count_center >= int(alpha * total_count):
        return center_value
    else:
        return 2


def sample_selection(mask, patch_size=7):
    # Generate determined sample for Identification of Changes

    final_labels = generic_filter(mask, classify_patch, size=patch_size)

    return final_labels
