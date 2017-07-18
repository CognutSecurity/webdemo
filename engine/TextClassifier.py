"""
QuickTemplate is the classifier class for tempalte suggestion in MyriadHub. It subclasses MyClassifier and is
instantiated with a proper preprocessor. The classifier uses SGD logistic regression for multi-class classification.

For more details on how to use SGD logistic regression and its corresponding parameters, please checkout:
http://scikit-learn.org/stable/modules/sgd.html

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

from Preprocessor import *
import numpy as np, sys
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from H3BaseClassifier import H3BaseClassifier


class TextClassifier(H3BaseClassifier):
   __name__ = 'TextClassifier'

   """
   TextClassifier main class: A Classifier used to classify text types
   """

   def __init__(self,
                loss='log',
                penalty='l2',
                l1_ratio=0,
                alpha=1e-4,
                shuffle=True,
                class_weight=None,
                **kwargs):

      """
      Constructor
      :return:
      """

      super(TextClassifier, self).__init__(**kwargs)
      # training params / optimal training params after model selection
      self.loss = loss
      self.penalty = penalty
      self.alpha = alpha
      self.shuffle = shuffle
      self.l1_ratio = l1_ratio
      if self.preprocessor is None:
         # the default preprocessor is a Tfidf vectorizer
         workflow_1 = [{'worker': 'TfidfVectorizer', 'params': {'encoding': 'utf-8'}}]
         self.preprocessor = [Preprocessor(workflow_1, ['n-grams'])]

      # default classifier is SGD logistic regressor
      self.classifier = SGDClassifier(loss=self.loss,
                                      penalty=self.penalty,
                                      alpha=self.alpha,
                                      l1_ratio=self.l1_ratio,
                                      shuffle=self.shuffle,
                                      class_weight=class_weight)

   def fit(self, X, y=None):
      """
      Training method for build the recommendation system model
      :param trainset: Nxd training data vectors
      """
      self.classifier.fit(X, y)
      self.status += 1
      # this only print logs
      super(TextClassifier, self).fit(X, y)

   def partial_fit(self, X, y=None, classes=None):
      """
      Partial training method to dynamically factor in new data into the current model
      :param partial_trainset: Data vectors for subset of data to be trained
      :param labels: Output vector for training data
      :param classes: Array of unique outputs
      :return:
      """

      self.classifier.partial_fit(X, y, classes=classes)
      self.status += 1
      super(TextClassifier, self).partial_fit(X, y)

   def predict(self, Xtt):
      """
      Classification of which template to use for replying an email
      :param testset: Mxd test data vectors
      :return: Template labels, e.g., 1,2,..., n (integer)
      """

      if self.status < 1:
         self.logger.error(__name__ + ' is not trained yet, exit!')
         sys.exit()
      super(TextClassifier, self).predict(Xtt)
      return self.classifier.predict(Xtt)

   def decision_function(self, Xtt):
      super(TextClassifier, self).decision_function(Xtt)
      return self.classifier.decision_function(Xtt)

   def plot_classifier(self, savepath=None, show=True):
      '''
      plotting ultilities for TextClassifier
      :param: savepath: the path to save the figure HTML files
      :return: NoneType
      '''

      from bokeh.plotting import output_file, figure, show, ColumnDataSource, save
      from bokeh.layouts import gridplot
      from bokeh.models import FixedTicker, HoverTool, BoxZoomTool, ResetTool, WheelZoomTool
      from sklearn.preprocessing import minmax_scale

      output_file(savepath, title='TextClassifier Summary')

      figs_row = []
      ticks = []
      tick_labels = []
      feature_idx = 0
      for p in self.preprocessor:
         if p._FEATURE_SIZE:
            if p._FEATURE_SIZE != len(p._FEATURE_NAMES):
               ticks.extend([p._FEATURE_SIZE / 2])
            else:
               ticks.extend(range(feature_idx, feature_idx + p._FEATURE_SIZE))
            feature_idx += p._FEATURE_SIZE
            tick_labels.extend(p._FEATURE_NAMES)
      # TOOLS = [HoverTool()]
      cls_id = 0
      for cls in self.classifier.classes_:
         cls_name = str(cls)
         if ticks[0] > 0:
            y_ticks = np.r_[np.array([self.classifier.coef_[cls_id][ticks[0]]]),
                            self.classifier.coef_[cls_id][ticks[1]:]].ravel()
         else:
            y_ticks = self.classifier.coef_[cls_id].ravel()
         y_ticks = minmax_scale(y_ticks, feature_range=(-1, 1))
         source = ColumnDataSource(data=dict(
            x=ticks,
            y=y_ticks / 2.0,
            v=y_ticks,
            names=tick_labels
         ))
         hover = HoverTool(tooltips=[("feature name", "@names"),
                                     ("value", "@v")])
         fig = figure(width=960, height=180,
                      title='Feature Relevance for Tempalte-ID: ' + cls_name,
                      tools=[hover, BoxZoomTool(), ResetTool(), WheelZoomTool()])
         fig.rect(x='x',
                  y='y',
                  width=.5,
                  height=y_ticks,
                  color='#CAB2D6',
                  source=source)

         fig.xaxis[0].ticker = FixedTicker(ticks=ticks)
         fig.xaxis.major_label_orientation = np.pi / 4
         figs_row.append([fig])
         cls_id += 1
      # show(gridplot(figs_row))
      save(gridplot(figs_row), savepath)


if __name__ == '__main__':
   """
   User Guide
   """

   from bokeh.layouts import gridplot, layout, column, row
   from bokeh.plotting import output_file, show, save, figure
   from sklearn.datasets import make_classification, load_digits, load_boston
   from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit
   from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
   from mlcore.utils.DataViz import DataViz

   # Load data
   # Normally we use Xtr denoting training feature set, and ytr denoting training labels
   # mydata = load_digits(4)
   # Xtr = mydata.data
   # ytr = mydata.target
   Xtr, ytr = make_classification(n_samples=1000, n_classes=3, n_features=10, n_informative=4, n_clusters_per_class=2)
   clf = TextClassifier()
   # split the dataset as train/test
   X_train, X_test, y_train, y_test = train_test_split(Xtr, ytr, test_size=0.2, random_state=30)

   epoch = 5
   batch_size = 20
   steps = 100
   tr_size = X_train.shape[0]

   accuracies = []
   precisions = []
   recalls = []
   for n in range(epoch):
      for i in xrange(steps):
         batch_start = i * batch_size % tr_size
         batch_end = min(batch_start + batch_size, tr_size)
         X_train_part = X_train[batch_start:batch_end]
         y_train_part = y_train[batch_start:batch_end]

         # Partial train the segment of data, classify test data and compare to actual values using various data analysis methods
         clf.partial_fit(X=X_train_part, y=y_train_part, classes=np.unique(ytr))
         y_pred = clf.predict(X_test)
         accuracy = accuracy_score(y_test, y_pred)
         f1 = f1_score(y_test, y_pred, average=None)
         precision = precision_score(y_test, y_pred, average='weighted')
         recall = recall_score(y_test, y_pred, average='weighted')

         accuracies.append(accuracy)
         precisions.append(precision)
         recalls.append(recall)

   cvfolds = ShuffleSplit(n_splits=5, test_size=0.2)
   tr_sizes, tr_scores, tt_scores = learning_curve(estimator=clf.classifier, n_jobs=1, X=Xtr, y=ytr, cv=3,
                                                   train_sizes=np.linspace(0.1, 1, 5), scoring='accuracy', verbose=1)

   # plotting
   config = {'colormap': 'Dark2_',
             'dot_size': 6,
             'line_width': 2,
             'width': 360,
             'height': 280,
             'output_file': 'multi.gaussians.html'}
   dv = DataViz(config)
   f1 = dv.feature_scatter1d(Xtr, names=[str(d) for d in np.arange(Xtr.shape[1])])
   f2 = dv.fill_between(tr_sizes, [tt_scores.mean(axis=-1), tr_scores.mean(axis=-1)],
                                  [tt_scores.std(axis=-1), tr_scores.std(axis=-1)], title='Accuracy Curves',
                        legend=['Test', 'Train'], xlim=[min(tr_sizes), max(tr_sizes)], ylim=[0, 1.05],
                        legend_orientation='horizontal')
   f3 = dv.project2d(Xtr, ytr, legend=['A', 'B'])
   fea_names = ['fea.'+str(idx+1) for idx in np.arange(Xtr.shape[1])]
   fea_names.append('target')
   f4 = dv.plot_corr(np.c_[Xtr, ytr], names=fea_names)
   f5 = dv.simple_curves(np.arange(epoch*steps), [accuracies, precisions, recalls],
                         legend=['Accuracy', 'Precision', 'Recall'],
                         xlim=[0, 20], title="SGD Training",
                         xlabel='Training steps', ylabel='Scores')
   plots = [f1, f2, f3, f4, f5]
   show(gridplot(plots, ncols=3))
   dv.send_to_server()