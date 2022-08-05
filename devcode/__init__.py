from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from devcode.models.local_learning import LocalModel
from devcode.utils import dummie2multilabel, cm2acc, scale_feat
from load_dataset import datasets


def run_simulation(dataset_name, kmeans, clf_model, test_size=0.2):
    X = datasets[dataset_name]['features'].values
    Y = datasets[dataset_name]['labels'].values

    # Train/Test split = 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    # Scaling features
    X_tr_norm, X_ts_norm = scale_feat(X_train, X_test, scaleType='min-max')

    lm = LocalModel(ClusterAlg=kmeans, ModelAlg=clf_model)
    lm.fit(X_tr_norm, y_train, verboses=1)

    y_pred_tr = lm.predict(X_tr_norm, rounded=True)
    y_pred_ts = lm.predict(X_ts_norm, rounded=True)

    cm_tr = confusion_matrix(dummie2multilabel(y_train),
                             dummie2multilabel(y_pred_tr))
    cm_ts = confusion_matrix(dummie2multilabel(y_test),
                             dummie2multilabel(y_pred_ts))

    acc_tr = cm2acc(cm_tr)
    acc_ts = cm2acc(cm_ts)

    print(f"Train accuracy: {acc_tr}\nTest accuracy:  {acc_ts}\n")