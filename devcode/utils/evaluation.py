import numpy as np


def per_round_metrics(confusion_matrix):
    """

    Parameters
    ----------
    confusion_matrix

    Returns
    -------

    """
    # data = cm_ts
    length  = confusion_matrix.shape[1]
    cm_side = int(np.sqrt(length))

    n_classes = len(confusion_matrix)

    accuracies    = [0] * n_classes
    specifities   = [0] * n_classes
    sensibilities = [0] * n_classes
    f1_scores     = [0] * n_classes

    for i in range(n_classes):
        cm_ith = np.reshape(confusion_matrix[i], (cm_side, cm_side))
        accuracies[i]    = cm2acc(cm_ith)
        sensibilities[i] = cm2results(cm_ith, cm2sen)
        specifities[i]   = cm2results(cm_ith, cm2esp)
        f1_scores[i]     = cm2results(cm_ith, cm2f1)

    return accuracies, specifities, sensibilities, f1_scores


def f_o(u):
    """ Objective function in validation strategy """
    return np.mean(u) - 2*np.std(u)


def cm2acc(cm):
    return np.trace(cm) / np.sum(cm)


def cm2results(cm, metric_func):
    n_classes = len(cm)
    if n_classes == 2:
        # Binary classification
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        return metric_func(tn, fp, fn, tp)
    else:
        # Multiclass classification
        tn, fp, fn, tp = [0] * n_classes, [0] * n_classes, [0] * n_classes, [0] * n_classes,
        metric_results = [None] * n_classes
        for i in range(n_classes):
            tn[i] = cm[i][i]
            fp[i] = np.sum(cm[i, :]) - tn
            fn[i] = np.sum(cm[:, i]) - tn
            tp[i] = np.trace(cm) - tn
            metric_results[i] = metric_func(tn[i], fp[i], fn[i], tp[i])

        return np.mean(metric_results)


def cm2sen(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn)
    return sensitivity


def cm2esp(tn, fp, fn, tp):
    specificity = tn / (tn + fp)
    return specificity


def cm2f1(tn, fp, fn, tp):
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
