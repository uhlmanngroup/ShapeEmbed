import tqdm
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, classification_report

def run_classification( X, y, n_splits=None
                      , logreg_maxiter=200 ):
  classifiers = {
    "LogisticRegression": LogisticRegression( solver='newton-cg'
                                            , max_iter=logreg_maxiter )
  , "RandomForest": RandomForestClassifier(random_state=42)
  }
  kf = StratifiedKFold(n_splits=n_splits if n_splits else len(np.unique(y)), shuffle=True, random_state=42)
  idx_split = list(kf.split(X, y))
  eval = { nm: evaluate_classifier(clf, nm, X, y, list(kf.split(X, y)))
           for nm, clf in classifiers.items()
         }
  return { nm: { 'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs)
               , 'micro_precision_mean': np.mean(mi_precs), 'micro_precision_std': np.std(mi_precs)
               , 'macro_precision_mean': np.mean(ma_precs), 'macro_precision_std': np.std(ma_precs)
               , 'weighted_precision_mean': np.mean(w_precs), 'weighted_precision_std': np.std(w_precs)
               , 'micro_recall_mean': np.mean(mi_recs), 'micro_recall_std': np.std(mi_recs)
               , 'macro_recall_mean': np.mean(ma_recs), 'macro_recall_std': np.std(ma_recs)
               , 'weighted_recall_mean': np.mean(w_recs), 'weighted_recall_std': np.std(w_recs)
               , 'micro_f1_mean': np.mean(mi_f1s), 'micro_f1_std': np.std(mi_f1s)
               , 'macro_f1_mean': np.mean(ma_f1s), 'macro_f1_std': np.std(ma_f1s)
               , 'weighted_f1_mean': np.mean(w_f1s), 'weighted_f1_std': np.std(w_f1s)
               , 'log_loss_mean': np.mean(loglosses), 'log_loss_std': np.std(loglosses) }
           for nm, (accs, mi_precs, ma_precs, w_precs, mi_recs, ma_recs, w_recs, mi_f1s, ma_f1s, w_f1s, loglosses, _, _) in eval.items() }

def evaluate_classifier(classifier, name, X, y, idx_splits):
  all_accuracy=[]
  micro_precision=[]
  macro_precision=[]
  weighted_precision=[]
  micro_recall=[]
  macro_recall=[]
  weighted_recall=[]
  micro_f1=[]
  macro_f1=[]
  weighted_f1=[]
  all_log_loss=[]
  all_y_test = []
  all_y_pred = []
  for train_index, test_index in (pbar:=tqdm.tqdm(idx_splits, desc=name, position=1, leave=True)):
    X_train, X_test = np.take(X, train_index, axis=0), np.take(X, test_index, axis=0)
    y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)
    classifier.fit(X_train, y_train)
    all_y_test.extend(y_test)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    all_y_pred.extend(y_pred)
    all_accuracy.append(accuracy_score(y_test, y_pred))
    micro_precision.append(precision_score(y_test, y_pred, average='micro'))
    macro_precision.append(precision_score(y_test, y_pred, average='macro'))
    weighted_precision.append(precision_score(y_test, y_pred, average='weighted'))
    micro_recall.append(recall_score(y_test, y_pred, average='micro'))
    macro_recall.append(recall_score(y_test, y_pred, average='macro'))
    weighted_recall.append(recall_score(y_test, y_pred, average='weighted'))
    micro_f1.append(f1_score(y_test, y_pred, average='micro'))
    macro_f1.append(f1_score(y_test, y_pred, average='macro'))
    weighted_f1.append(f1_score(y_test, y_pred, average='weighted'))
    all_log_loss.append(log_loss(y_test, y_prob, labels=classifier.classes_))
    pbar.update()
  rpt = classification_report(all_y_test, all_y_pred)
  cm = confusion_matrix(all_y_test, all_y_pred)
  return all_accuracy, micro_precision, macro_precision, weighted_precision, micro_recall, macro_recall, weighted_recall, micro_f1, macro_f1, weighted_f1, all_log_loss, rpt, cm
