from glob import glob
import numpy as np
import pandas as pd
#from evaluate_plot import pick_plot, recall_precsion

def find_min_abs_original_value(lst):
    min_abs_value = abs(lst[0])
    min_original_value = lst[0]

    for num in lst:
        if abs(num) < min_abs_value:
            min_abs_value = abs(num)
            min_original_value = num

    return min_original_value

def return_double_diff(lst1, lst2):
    lst = lst1 + lst2
    mean_abs = np.mean([abs(i) for i in lst])
    var_abs = np.var(lst)
    mean = np.mean(lst)
    return mean_abs, var_abs, '%.3f'%(mean)

def return_single_diff(lst):
    mean_abs = np.mean([abs(i) for i in lst])
    var_abs = np.var(lst)
    mean = np.mean(lst)
    return mean_abs, var_abs, '%.3f'%(mean)


def get_recall_precison_result(path, prob_threshold, sampling_rate):
    p_manual_sum, s_manual_sum = 0, 0
    p_ai_sum, s_ai_sum = 0, 0
    p_diff, s_diff = [], []

    for i in glob(path + '/*'):
        p_manual_sum += 1
        s_manual_sum += 1
        _, p_manual, s_manual = i.split('/')[-1].split('_')[:3]
        p_manual = float(p_manual)
        s_manual = float(s_manual)
        p_result = []
        s_result = []
        with open(i, 'r') as f:
            for line in f:
                lines = line.strip().split()
                ai_time, prob, type = lines[:3]
                ai_time = float(ai_time)
                prob = float(prob)
                if type == 'P':
                    if prob >= prob_threshold:
                        p_ai_sum += 1
                        if abs(float(ai_time) - float(p_manual)) <= sampling_rate:
                            p_result.append(float(ai_time) - float(p_manual))
                else:
                    if prob >= prob_threshold:
                        s_ai_sum += 1
                        if abs(float(ai_time) - float(s_manual)) <= sampling_rate:
                            s_result.append(float(ai_time) - float(s_manual))

        if p_result:
            p_diff.append(find_min_abs_original_value(p_result)/sampling_rate)

        if s_result:
            s_diff.append(find_min_abs_original_value(s_result)/sampling_rate)
    if p_ai_sum == 0:
        p_precison = 1
    else:
        p_precison = len(p_diff) / p_ai_sum
    if s_ai_sum == 0:
        s_precision = 1
    else:
        s_precision = len(s_diff) / s_ai_sum
    return len(p_diff)/p_manual_sum, len(s_diff)/s_manual_sum, p_precison, s_precision, p_diff, s_diff

def main(path1, path2):
    prob_index, abs_index_eqt, abs_index_pht, index_eqt, index_pht, sqrt_index_eqt, sqrt_index_pht = [], [], [], [], [], [], []
    rec_pre_index, p_rec_eqt, p_rec_pht, s_rec_eqt, s_rec_pht = [], [], [], [], []
    p_pre_eqt, s_pre_eqt, p_pre_pht, s_pre_pht = [], [], [], []
    for i in range(30, 90):
        prob_threshold = i / 100
        rec_pre_index.append(prob_threshold)
        p_rec1, s_rec1, p_pre1, s_pre1, p_diff1, s_diff1 = get_recall_precison_result(path1, prob_threshold)
        p_rec2, s_rec2, p_pre2, s_pre2, p_diff2, s_diff2 = get_recall_precison_result(path2, prob_threshold)
        p_rec_eqt.append(p_rec1)
        s_rec_eqt.append(s_rec1)
        p_pre_eqt.append(p_pre1)
        s_pre_eqt.append(s_pre1)
        p_rec_pht.append(p_rec2)
        s_rec_pht.append(s_rec2)
        p_pre_pht.append(p_pre2)
        s_pre_pht.append(s_pre2)
        if i % 10 == 0:
            prob_index.append(prob_threshold)
            mean_abs1, var_abs1, mean1 = return_double_diff(p_diff1, s_diff1)
            mean_abs2, var_abs2, mean2 = return_double_diff(p_diff2, s_diff2)
            abs_index_eqt.append(mean_abs1)
            abs_index_pht.append(mean_abs2)
            index_eqt.append(mean1)
            index_pht.append(mean2)
            sqrt_index_eqt.append(var_abs1)
            sqrt_index_pht.append(var_abs2)

    pick_plot(prob_index, abs_index_eqt, abs_index_pht, index_eqt, index_pht, sqrt_index_eqt, sqrt_index_pht)
    recall_precsion(p_rec_eqt, s_rec_eqt, p_rec_pht, s_rec_pht, p_pre_eqt, s_pre_eqt, p_pre_pht, s_pre_pht)

def main_dispersed(**kwargs):
    for name, path in kwargs.items():
        print(f'Name:{name} Path:{path}')
        #for i in [0.5, 0.3, 0.1]:
        for i in [0.1]:
            print(f'Prob threshold: {i}')
            p_rec, s_rec, p_pre, s_pre, p_diff, s_diff = get_recall_precison_result(path[0], i, path[1])
            print('P')
            print(f'recall:{p_rec*100:.2f}% precison:{p_pre*100:.2f}%')
            mean_abs, var_abs, mean = return_single_diff(p_diff)
            print(f'Mean: {-float(mean)}s, Var: {np.sqrt(var_abs):.4f}s, mean_abs: {mean_abs}s')
            print('S')
            print(f'recall:{s_rec*100:.2f}% precison:{s_pre*100:.2f}%')
            mean_abs, var_abs, mean = return_single_diff(s_diff)
            print(f'Mean: {-float(mean)}s, Var: {np.sqrt(var_abs):.4f}s, mean_abs: {mean_abs}s')


if __name__ == '__main__':
    main_dispersed(original=['original_model', 100],
                   tf = ['transfer_learning', 100],
                   swag = ['bht_test_dataset', 50])






