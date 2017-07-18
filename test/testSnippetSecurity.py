'''
A Java snippet security detector using TfIdf
'''
import re, csv, datetime as dt, argparse
import os.path, dill
import numpy as np
from h3db.model.JavaSnippets import *
from mlcore.engine.TextClassifier import TextClassifier
from mlcore.engine.Preprocessor import Preprocessor
from mlcore.utils.plot_utils import print_confmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, validation_curve, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
from sklearn.decomposition import PCA
from bokeh.plotting import figure, output_file, save, show

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

# initialize command parser
argparser = argparse.ArgumentParser()
argparser.add_argument("--filename", help="input snippet csv file path...")
argparser.add_argument("--xvmodel", action='store_true', help="cross validation model selection...")
argparser.add_argument("--binary", action='store_true', help="binary classes: security-relevant(1-4) or nonsecurity(5)")
argparser.add_argument("-f", "--nfolds", help="(optional) num. of folds for cross validation, default: 5", default=5, type=int)
args = argparser.parse_args()

raw_data_path = args.filename
xvmodel = args.xvmodel
n_folds = args.nfolds
is_binary = args.binary

datafile = ''.join(raw_data_path.split('.')[:-1]) + '.p'
if os.path.isfile(datafile):
   dataset = dill.load(open(datafile, 'rb'))
   Xtr = dataset['Xtr']
   ytr = dataset['ytr']
   hash_id = dataset['hash']
else:
   # initialize database
   db = JavaSnippets._meta.database
   db.connect()
   db.create_tables([JavaSnippets], safe=True)
   JavaSnippets.delete().execute()
   print 'database is cleared... '
   # parse raw data
   with open(raw_data_path, 'rb') as csv_file:
      samples = csv.reader(csv_file, delimiter=',')
      X, y, hash_id = list(), list(), list()
      for id, sample in enumerate(samples):
         sample[0] = remove_comments(sample[0])
         X.append(sample[0])
         if sample[1]:
            y.append(int(sample[1]))
         else:
            y.append(5)
         hash_id.append(sample[3])
         if id > 0 and id % 100 == 0:
            print '{:d} samples are processed ... '.format(id)
         if JavaSnippets.select().where(JavaSnippets.snippet_id == id).count() == 0:
            # insert new row
            snippet = JavaSnippets()
            snippet.snippet_id = id
            snippet.snippet = sample[0]
            snippet.true_sec_level = y[-1]
            snippet.save()
      work_flow = [{'worker': 'TfidfVectorizer', 'params': {'encoding': 'utf-8', 'min_df': 0.001, 'max_df': 0.99}},
                   {'worker': 'FeatureScaler', 'params': {'type': 'minmax'}}]
      pp = Preprocessor(work_flow)
      Xtr = pp.run(X)
      ytr = np.array([-1 if j != 4 else 1 for j in y] if is_binary else y)
      vocab = dict()
      for k, v in pp._PIPELINE[0]._worker.vocabulary_.items():
         vocab[v] = k
      dill.dump({'Xtr': Xtr, 'ytr': ytr, 'hash': hash_id, 'vocab': vocab}, open(datafile, 'w+'))

# prepare classifier
# SnippetClf = QuickTemplate(preprocessor=[pp], loss='hinge', penalty='l2')
svc = SVC(C=.5, kernel='linear', class_weight='balanced')
clf = svc
# print 'label 4: {:d}/{:d}'.format(np.where(np.array(y)==4)[0].size, len(y))
# pca = PCA(n_components=15)
# Xtr = pca.fit_transform(Xtr)

if xvmodel:
   print 'grid search model selection ... '
   clf_optimal = GridSearchCV(estimator=clf, param_grid={'C': np.linspace(0.1, 5, num=10)}, scoring='precision')
   clf_optimal.fit(Xtr, ytr)
   print 'best C: {:2.3f}, best score: {:2.3f}'.format(clf_optimal.best_params_['C'], clf_optimal.best_score_)

else:

   if n_folds > 1:
      # do cross validation
      cvfolds = StratifiedKFold(n_splits=n_folds, shuffle=True)
      acc_scores, prec_scores = list(), list()
      fold_id = 0
      for tr_idx, tt_idx in cvfolds.split(Xtr, ytr):
         xtr_fold, xtt_fold = Xtr[tr_idx], Xtr[tt_idx]
         ytr_fold, ytt_fold = ytr[tr_idx], ytr[tt_idx]
         print '\n[{:d}]-fold: {:d} training samples / {:d} testing samples'.format(fold_id, len(tr_idx), len(tt_idx))
         run_start = dt.datetime.now()
         svc.fit(xtr_fold, ytr_fold)
         y_predict = svc.predict(xtt_fold)
         acc = accuracy_score(ytt_fold, y_predict)
         prec = precision_score(ytt_fold, y_predict, average='binary', pos_label=1)
         acc_scores.append(acc)
         prec_scores.append(prec)
         run_end = dt.datetime.now()
         print print_confmat(ytt_fold, y_predict)
         print 'Accuracy: {:f}, Precision: {:f}, time spent {:f} sec.'.format(acc, prec, (run_end - run_start).total_seconds())
         print 'class weights: ', svc.class_weight_
         fold_id += 1
         for elem in zip(tt_idx, y_predict):
            tid, yt = elem[0], elem[1]
            query = JavaSnippets.update(predict_sec_level=yt).where(JavaSnippets.snippet_id == tid)
            query.execute()

      # final stats
      print 'Avg. accuracy on {:d} folds cross validation: {:2.3f}'.format(n_folds, np.array(acc_scores).mean())
      print 'Avg. precision on {:d} folds cross validation: {:2.3f}'.format(n_folds, np.array(prec_scores).mean())
   elif n_folds == 1:
      # LOO
      loo_res = np.zeros(ytr.size)
      print '\nLOO Evaludation on {:d} samples ...'.format(ytr.size)
      fresh_start_at = dt.datetime.now()
      for idx, elem in enumerate(loo_res):
         tt_idx = np.array([idx])
         tr_idx = np.setdiff1d(np.arange(ytr.size), np.array([idx]))
         run_start = dt.datetime.now()
         svc.fit(Xtr[tr_idx], ytr[tr_idx])
         loo_res[idx] = svc.predict(Xtr[tt_idx])[0]
         run_end = dt.datetime.now()
         print '{:d}/{:d} iterations done... [{:d}->{:d}] {:f} seconds elapsed ... '.\
            format(idx, ytr.size, int(ytr[idx]), int(loo_res[idx]), (run_end - run_start).total_seconds())

      acc = accuracy_score(ytr, loo_res)
      prec = precision_score(ytr, loo_res, average='binary', pos_label=1)
      all_done_at = dt.datetime.now()
      print print_confmat(ytr, loo_res)
      print 'Accuracy: {:f}, Precision: {:f}, time spent {:f} sec.'.format(acc, prec, (all_done_at - fresh_start_at).total_seconds())
