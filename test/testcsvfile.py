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
argparser.add_argument("inputfile", help="input test snippet csv file path...")
argparser.add_argument("-o", "--output", help="output csv file", type=str, default='test_results.csv')
args = argparser.parse_args()

input_file = args.inputfile
output_file = args.output

trainfile = '/Users/hxiao/repos/AndroidCP/ml/data/train/answer_snippets_coded.csv'

clf = SVC(C=.644, kernel='linear', class_weight='balanced')
work_flow = [{'worker': 'TfidfVectorizer', 'params': {'encoding': 'utf-8', 'min_df': 0.001, 'max_df': 0.99}},
                {'worker': 'FeatureScaler', 'params': {'type': 'minmax'}}]
pp = Preprocessor(work_flow)
# parse raw data
with open(trainfile, 'rb') as csv_file:
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
   Xtr = pp.run(X)
   ytr = np.array([-1 if j != 4 else 1 for j in y])
   print 'training ... '
   clf.fit(Xtr, ytr)

with open(input_file, 'rb') as input_csv:
   samples = csv.reader(input_csv, delimiter=',')
   X, hash_id = list(), list()
   for id, sample in enumerate(samples):
      sample[0] = remove_comments(sample[0])
      X.append(sample[0])
      hash_id.append(sample[-1])
   Xtt = pp.run(X, restart=False)
   print 'testing ' + input_file + '...'
   y_predict = clf.predict(Xtt)

with open(output_file, 'w+') as output_csv:
   print 'write back results in ' + output_file + ' ...'
   for hash, ytt in zip(hash_id, y_predict):
      output_csv.write(hash + ',' + str(ytt) + '\r\n')





