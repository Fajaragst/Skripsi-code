from sklearn import metrics
from sklearn.metrics import confusion_matrix    #Confussion Matrix

import math

def bal(y_true, Y_pred):
#     repro()
    tn, fp, fn, tp = confusion_matrix(y_true, Y_pred, labels=[0,1]).ravel()
    pf = fp/(fp+tn)
    bal = 1 - ((math.sqrt((1-recall(y_true, Y_pred))**2+ pf**2))/(math.sqrt(2)))
    return bal

def auc(y_test, Y_pred):
#     repro()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_pred, pos_label=True)
    auc = metrics.roc_auc_score(y_test, Y_pred)
    return auc, fpr, tpr

def gmeans(y_true, Y_pred):
    # repro()
    tn, fp, fn, tp = confusion_matrix(y_true, Y_pred, labels=[0,1]).ravel()
    pf = fp/(fp+tn)
    pd = tp/(tp+fn)
    gmeans = math.sqrt(pd*(1-pf))
    return gmeans

def pf(y_true, Y_pred):
#     repro()
    tn, fp, fn, tp = confusion_matrix(y_true, Y_pred, labels=[0,1]).ravel()
    # print(tn)
    pf = fp/(fp+tn)
    return pf

def recall(y_true, Y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, Y_pred, labels=[0,1]).ravel()
    pd = tp/(tp+fn)
    return pd

def conf_matrix(y_true, Y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, Y_pred, labels=[0,1]).ravel()
    return tn,fp,fn,tp