#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNASolventAccessibility -> feature_extract
@IDE    ：PyCharm
@Date   ：2021/5/9 15:54
=================================================='''
import os, torch, random
import numpy as np
from numba import jit
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()


class OneHotPSFMSSGetWindowPadheadfoot():

    def __init__(self, nucle_name: str, result_dir, win_size=25):
        super(OneHotPSFMSSGetWindowPadheadfoot, self).__init__()
        self.nucle_name = nucle_name
        self.fa_path = os.path.join(result_dir, nucle_name + ".fa")
        self.result_dir = result_dir
        self.psfm_path = os.path.join(self.result_dir, self.nucle_name + ".psfm")
        self.ss_path = os.path.join(self.result_dir, self.nucle_name + ".ss")
        self.onehot_path = os.path.join(self.result_dir, self.nucle_name)
        self.win_size = win_size
        self.stride = int(win_size / 2)

    @jit
    def getIthProteinLen(self):
        seq = np.loadtxt(self.fa_path, dtype=str)[1]
        nucle_length = len(seq)
        return nucle_length

    @jit
    def feature(self):
        nucle_length = self.getIthProteinLen()
        one_hot = np.loadtxt(self.onehot_path, dtype=float)
        psfm = np.loadtxt(self.psfm_path, dtype=float)
        ss = np.expand_dims(np.loadtxt(self.ss_path, dtype=float), 1)

        one_hot_psfm_ss = np.append(one_hot, psfm, axis=1)
        one_hot_psfm_ss = np.append(one_hot_psfm_ss, ss, axis=1)

        nucle_length, fea_num_one_hot = one_hot.shape
        paddingheader = one_hot[:self.stride, :]
        paddingfooter = one_hot[-self.stride:, :]
        one_hot = np.append(paddingheader, one_hot, axis=0)
        one_hot = np.append(one_hot, paddingfooter, axis=0)

        nucle_length, fea_num_psfm = psfm.shape
        paddingheader = psfm[:self.stride, :]
        paddingfooter = psfm[-self.stride:, :]
        psfm = np.append(paddingheader, psfm, axis=0)
        psfm = np.append(psfm, paddingfooter, axis=0)

        nucle_length, fea_num_ss = ss.shape
        paddingheader = ss[:self.stride, :]
        paddingfooter = ss[-self.stride:, :]
        ss = np.append(paddingheader, ss, axis=0)
        ss = np.append(ss, paddingfooter, axis=0)

        feature_one_hot = np.zeros((nucle_length, self.win_size * fea_num_one_hot))
        feature_psfm = np.zeros((nucle_length, self.win_size * fea_num_psfm))
        feature_ss = np.zeros((nucle_length, self.win_size * fea_num_ss))

        feature_one_hot_reverse = np.zeros((nucle_length, self.win_size * fea_num_one_hot))
        feature_psfm_reverse = np.zeros((nucle_length, self.win_size * fea_num_psfm))
        feature_ss_reverse = np.zeros((nucle_length, self.win_size * fea_num_ss))

        for i in range(self.stride, nucle_length + self.stride):
            feature_one_hot[i - self.stride, :] = one_hot[i - self.stride:i + self.stride + 1, :].flatten()
            feature_psfm[i - self.stride, :] = psfm[i - self.stride:i + self.stride + 1, :].flatten()
            feature_ss[i - self.stride, :] = ss[i - self.stride:i + self.stride + 1, :].flatten()

            feature_one_hot_reverse[i - self.stride, :] = one_hot[i - self.stride:i + self.stride + 1, :].flatten()
            feature_psfm_reverse[i - self.stride, :] = psfm[i - self.stride:i + self.stride + 1, :].flatten()
            feature_ss_reverse[i - self.stride, :] = ss[i - self.stride:i + self.stride + 1, :].flatten()

        feature_one_hot_reverse, feature_psfm_reverse, feature_ss_reverse = np.fliplr(
            feature_one_hot_reverse), np.fliplr(feature_psfm_reverse), np.fliplr(feature_ss_reverse)
        feature_one_hot_reverse, feature_psfm_reverse, feature_ss_reverse = np.ascontiguousarray(
            feature_one_hot_reverse), np.ascontiguousarray(feature_psfm_reverse), np.ascontiguousarray(
            feature_ss_reverse)
        one_hot_psfm_ss_reverse = np.flip(one_hot_psfm_ss, axis=1)
        one_hot_psfm_ss_reverse = np.ascontiguousarray(one_hot_psfm_ss_reverse)
        sample = {'fea': (feature_one_hot, feature_psfm, feature_ss, one_hot_psfm_ss), 'fea_reverse': (
            feature_one_hot_reverse, feature_psfm_reverse, feature_ss_reverse, one_hot_psfm_ss_reverse),
                  'nucleotide': self.nucle_name}  # construct the dictionary
        return sample

    @jit
    def getIthSampleFea(self):
        sample = self.feature()
        fea = sample['fea']
        fea_reverse = sample['fea_reverse']
        nucleotide = sample['nucleotide']
        return fea, fea_reverse, nucleotide
