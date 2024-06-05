from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score


def cacl_metrics(y_t, y_p):
    dic = dict()
    dic['Accuracy'] = accuracy_score(y_t, y_p)
    dic['Precision'] = average_precision_score(y_t, y_p)
    dic['F1Score'] = f1_score(y_t, y_p)
    dic['Recall'] = recall_score(y_t, y_p)
    return dic
