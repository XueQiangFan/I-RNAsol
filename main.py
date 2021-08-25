#!/Users/11834/.conda/envs/Pytorch_GPU/python.exe
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File    ：RNASolventAccessibility -> main
@IDE    ：PyCharm
@Date   ：2021/5/9 15:53
=================================================='''
import os
import numpy as np
import torch
from I_RNAsol_webserver.Util.feature_extract import OneHotPSFMSSGetWindowPadheadfoot
from I_RNAsol_webserver.MVCADNN import BiLSTM_SE_Net
from I_RNAsol_webserver.Util.WriteFile import appendWrite
from I_RNAsol_webserver.Util.GEN_HTML import GEN_HTML
from I_RNAsol_webserver.Util.feature_generate import FeaturesGeneration
import warnings

warnings.filterwarnings('ignore')


def MAXASAValue(residue_type):
    nucle_ASA = {"A": 400,
                 "C": 350,
                 "G": 400,
                 "U": 350}
    return nucle_ASA[residue_type]


def tester(nucle_name, result_dir):
    fa_path = os.path.join(result_dir, nucle_name)
    save_model = "./save_model/"
    model = BiLSTM_SE_Net()
    saved_model = save_model + 'epoch_' + str(2709)
    model.load_state_dict(torch.load(saved_model, map_location="cpu"))
    optimizer = torch.optim.Adam(model.parameters())
    saved_model = save_model + 'epoch_' + str(2709) + 'opt'
    optimizer.load_state_dict(torch.load(saved_model, map_location="cpu"))

    model.eval()
    with torch.no_grad():
        Data = OneHotPSFMSSGetWindowPadheadfoot(nucle_name, result_dir)
        fea, fea_reverse, nucleotide = Data.getIthSampleFea()
        feature_one_hot, feature_psfm, feature_ss, one_hot_psfm_ss = torch.FloatTensor(fea[0]), torch.FloatTensor(
            fea[1]), torch.FloatTensor(fea[2]), torch.FloatTensor(fea[3])
        feature_one_hot, feature_psfm, feature_ss, one_hot_psfm_ss = torch.unsqueeze(feature_one_hot,
                                                                                     0), torch.unsqueeze(feature_psfm,
                                                                                                         0), torch.unsqueeze(
            feature_ss, 0), torch.unsqueeze(one_hot_psfm_ss, 0)
        feature_one_hot_rev, feature_psfm_rev, feature_ss_rev, one_hot_psfm_ss_rev = torch.FloatTensor(
            fea_reverse[0]), torch.FloatTensor(
            fea_reverse[1]), torch.FloatTensor(fea_reverse[2]), torch.FloatTensor(fea_reverse[3])
        feature_one_hot_rev, feature_psfm_rev, feature_ss_rev, one_hot_psfm_ss_rev = torch.unsqueeze(
            feature_one_hot_rev, 0), torch.unsqueeze(
            feature_psfm_rev, 0), torch.unsqueeze(feature_ss_rev, 0), torch.unsqueeze(one_hot_psfm_ss_rev, 0)

        predict00 = model(feature_one_hot, feature_psfm, one_hot_psfm_ss, feature_ss)
        predict01 = model(feature_one_hot_rev, feature_psfm_rev, one_hot_psfm_ss_rev, feature_ss_rev)
        predict = (predict00[4] + predict01[4]) / 2

        seq = np.loadtxt(fa_path, dtype=str)[1]
        nucle_length = len(seq)
        filename = nucleotide + ".sa"
        file_path = os.path.join(result_dir, filename)
        if os.path.exists(file_path):
            pass
        else:
            appendWrite(file_path, '{:>4}\n\n'.format("# I-RNAsol VFORMAT (I-RNAsol V1.0)"))
            appendWrite(file_path, '{:>1}  {:>1}  {:>4}  {:>4}\t\n'.format("NO.", "AA", "RSA", "ASA"))
            for i in range(nucle_length):
                index, residue, RSA = i + 1, seq[i], predict[i, 0]
                SA = MAXASAValue(seq[i]) * predict[i, 0]
                appendWrite(file_path, '{:>4}  {:>1}  {:>.3f}  {:>.3f}\t\n'.format(index, residue, RSA, SA))
            appendWrite(file_path, '{:>8} \t'.format("END"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="I-RNAsol Predicts RNA Solvent Accessibility")
    parser.add_argument("-n", "--nucle_name", required=True, type=str, help="nucleotide name")
    parser.add_argument("-s", "--sequence", required=True, type=str, help="AA sequence ")
    parser.add_argument("-o", "--result_path", required=True, type=str, help="save result path")
    args = parser.parse_args()
    features_generation = FeaturesGeneration(args.nucle_name, args.sequence, args.result_path)
    features_generation.One_Hot_Encoding()
    features_generation.LinearParitition_SS()
    features_generation.PSFM_generation()
    tester(args.nucle_name, args.result_path)
    gan_html = GEN_HTML(args.nucle_name, args.result_path)
    gan_html.generate_html()


if __name__ == '__main__':
    main()
