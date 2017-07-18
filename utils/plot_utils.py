"""
This utility file contains helper functions for facilitating plotting

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import numpy as np

def getBoxbyX(X, grid=50, padding=True):
    '''
    Get the meshgrid X,Y given data set X
    :param X: dataset
    :param grid: meshgrid step size
    :param padding: if add extra padding around
    :return: X,Y
    '''
    if X.shape[1] > 2:
        print 'We can only get the grid in 2-d dimension!'
        return None
    else:
        minx = min(X[:,0])
        maxx = max(X[:,0])
        miny = min(X[:,1])
        maxy = max(X[:,1])
        padding_x = 0.05*(maxx-minx)
        padding_y = 0.05*(maxy-miny)
        if padding:
            X,Y = np.meshgrid(np.linspace(minx-padding_x, maxx+padding_x, grid),
                              np.linspace(miny-padding_y, maxy+padding_y, grid))
        else:
            X,Y = np.meshgrid(np.linspace(minx, maxx, grid),
                              np.linspace(miny, maxy, grid))
    return (X, Y)

def setAxSquare(ax):
    xlim0, xlim1 = ax.get_xlim()
    ylim0, ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1-xlim0)/(ylim1-ylim0))


def print_confmat(true_labels, predicted_labels, label_names=None):
    '''
    Print out confusion matrix out of true labels and predicted ones
    :param true_labels: as name suggested
    :param predicted_labels: as name suggested
    :param label_names: a dict of label names, key->name
    :return: a output string
    '''

    import sys
    output_str = "[Confusion Matrix]"
    if label_names is not None:
        n_labels = len(label_names.keys())
    else:
        n_labels = true_labels.size
        if true_labels.size != predicted_labels.size:
            sys.exit('Dimension of labels inconsisitent! exit!')
        else:
            uniq_labels = np.unique(true_labels)

    header_line = "\n{:15s}".format("true/predict")
    res_lines = []
    for label in uniq_labels:
        label_name = label_names[label] if label_names and label_names.has_key(label) else str(label)
        header_line += "{:10s}".format('('+label_name+')')
        res_line = "\n{:15s}".format('('+label_name+')')
        predicted = predicted_labels[np.where(true_labels==label)[0]]
        for prd_label in uniq_labels:
            res_line += "{:10s}".format(' ' + str(np.where(predicted == prd_label)[0].size) + ' ')
        res_lines.append(res_line)

    output_str += header_line
    for ln in res_lines:
        output_str += ln
    return output_str


