import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



def F_reduce_time(time_v, value_m, L=10, S=5):
    """
    inputs:
        - time_v (nb_frame)
        - value_m (nb_dim, nb_frame)
        - L
        - S
    """
    nb_frame = value_m.shape[1]

    time_l = []
    value_l = []
    num_frame = 0
    while num_frame+L < nb_frame:
        time_l.append(np.mean(time_v[num_frame:num_frame+L]))
        m_v = np.mean(value_m[:,num_frame:num_frame+L], axis=1)
        s_v = np.std(value_m[:,num_frame:num_frame+L], axis=1)
        value_l.append( np.concatenate((m_v,s_v), axis=0) )
        num_frame += S
    time_v = np.asarray(time_l).T
    value_m = np.asarray(value_l).T
    
    return time_v, value_m



def F_stretch_ssm(SSM_m, do_affiche=True):
    R = SSM_m.flatten()

    [value_v, bins_v] = np.histogram(R, bins=500)
    proba_v = value_v / np.sum(value_v)
    cumul_proba_v = np.cumsum(proba_v)
    
    newSSM_m = np.zeros(SSM_m.shape)

    step = 0.05
    target_proba_v = np.arange(0.0, 1.0+step, step)
    for idx in range(len(target_proba_v)-1):
        s = target_proba_v[idx]
        e = target_proba_v[idx+1]
        result_v = bins_v[ np.where((s <= cumul_proba_v) & (cumul_proba_v <= e))[0]+1 ]
        if len(result_v):
            r1 = result_v[0]
            r2 = result_v[-1]
            #print(s, e, ' -> ', r1, r2, ' -> ', 0.5*(s+e))
            newSSM_m[(r1 <= SSM_m) & (SSM_m <= r2)] = 0.5*(s+e)

    if do_affiche:
        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.imshow(SSM_m, aspect='equal', cmap=cm.inferno); #plt.colorbar(orientation='vertical')
        plt.subplot(1,2,2)
        plt.imshow(newSSM_m, aspect='equal', cmap=cm.inferno); #plt.colorbar(orientation='vertical')
    
    return newSSM_m



def F_get_structure_annot(file):
    """
    description:
        - load a SALAMI ground-truth annotation file for MSD
    inputs:
        - file: file in .csv format
    outputs:
        - annot_l: list of segments, each as a tuple of (start-segment, segment-label)
    """
    with open(file) as fid:
        all_lines_l = fid.readlines()
    annot_l = [(float(one_line.split('\t')[0]), one_line.split('\t')[1].strip('\n')) for one_line in all_lines_l ]
    return annot_l



def F_boundaries_to_segments(boudaries_l):
    """
    description:
        convert segment boundaries position into segments start and end time
    inputs:
        - boundaries_l (nb_boundaries) gives the position of each segment boundaries
    outputs:
        - segment_m (nb_segment, 2) gives the start and end time of each segment
    """
    segment_m = np.zeros((len(boudaries_l)-1,2))
    for idx in range(len(boudaries_l)-1):
        segment_m[idx,0] = boudaries_l[idx]
        segment_m[idx,1] = boudaries_l[idx+1]
    return segment_m
