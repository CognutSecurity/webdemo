__author__ = 'morgan'
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save, show
from bokeh.charts import Scatter
from bokeh.models import ColumnDataSource, LabelSet
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE
import getopt, sys, os.path
import mlcore.utils.plot_utils as pltool
import importlib

class DatasetViewer(object):
   '''
   A class for dataset viewer used to analyse features in data
   set straightforwardly
   '''

   def __init__(self):
      pass

   def view_features_distribution(self, X):
      n, d = X.shape
      indices = []
      for i in np.arange(d):
         indices.extend((i + 1) * np.ones(n))
      plt.gca().scatter(np.asarray(indices), X.T.flatten(),
                        s=5, marker='.', alpha=0.5, facecolor='None', edgecolor='k')
      plt.gca().plot([1, d], [0, 0], 'k-')
      plt.gca().set(xlabel='feature id',
                    ylabel='feature value',
                    title='Feature distribution',
                    xlim=[0, d - 1])

   def project2d(self, X, y=None, method='pca', g=0.5):
      '''
      Project high-dimensiona data to 2D for visulaization
      using methods e.g., pca, kernel-pca, mds
      :param X: N*d dataset
      :param y: labels/None if not given
      :param method: string in ['pca','kpca','mds']
      :return: projected dataset X_project
      '''
      fig = figure(title="Data Projection on 2D after dimension reduction by "+method, webgl=True,
                   plot_width=883, plot_height=615, toolbar_location='above')
      dot_size = 0.02
      if y is not None and y.size != X.shape[0]:
         exit("Data dims are not matched!")
      else:
         n_comp = 2
         if method == 'pca':
            projector = PCA(n_components=n_comp)
            X_proj = projector.fit_transform(X)
         elif method == 'kpca':
            projector = KernelPCA(n_components=n_comp, kernel='rbf', gamma=g)
            X_proj = projector.fit_transform(X)
         elif method == 'mds':
            projector = MDS(n_components=n_comp)
            X_proj = projector.fit_transform(X)
         elif method == 'tsne':
            projector = TSNE(n_components=n_comp)
            X_proj = projector.fit_transform(np.array(X, dtype='float'))
         else:
            print 'No projector found!'
            X_proj = X

      if y is not None:
         n_labels = np.unique(y).size
         if n_labels > 2:
            cmap = getattr(importlib.import_module('bokeh.palettes'), 'Spectral'+str(n_labels))
         else:
            cmap = ['red', 'blue']
         dsource = ColumnDataSource(data=dict(
                                       xcoord=X_proj[:,0],
                                       ycoord=X_proj[:,1],
                                       labels=y,
                                       colors=[cmap[np.where(np.unique(y) == i)[0]] for i in y]))
         # plot_labels = LabelSet(x='xcoord', y='ycoord', text='labels',
         #                        source=dsource, level='glyph', render_mode='canvas')
         fig.circle(x='xcoord', y='ycoord', color='colors', legend='labels', radius=dot_size, source=dsource)
         # fig.add_layout(plot_labels)
      else:
         dsource = ColumnDataSource(data=dict(xcoord=X_proj[:, 0], ycoord=X_proj[:, 1]))
         fig.circle(x='xcoord', y='ycoord', radius=dot_size, source=dsource)
      # output the plot in HTML
      output_file('temp_plotting.html', title='projection in 2D')
      show(fig)

   def plot_corr(self, X, names=None):
      n, d = X.shape
      xcorr = np.corrcoef(X.T)
      XX, YY = np.meshgrid(np.arange(d), np.arange(d))
      a1 = plt.scatter(XX.ravel(), YY.ravel(),
                       s=15,
                       c=xcorr.ravel(), cmap='jet', alpha=0.7)
      plt.gca().set(title='Correlations of features',
                    xlim=[0, d - 1],
                    ylim=[0, d - 1])
      plt.colorbar(ax=plt.gca())
      if names is not None and len(names) == d:
         a1.set_xticklabels(names, rotation=90)
         a1.set_yticklabels(names)

      pltool.setAxSquare(plt.gca())

   def visualize(self, X, y, names=None):
      plt.subplot(2, 2, 1)
      self.view_features_distribution(X)
      pltool.setAxSquare(plt.gca())
      plt.subplot(2, 2, 2)
      self.project2d(X, y, method='kpca')
      pltool.setAxSquare(plt.gca())
      plt.subplot(2, 2, 3)
      self.project2d(X, y, method='pca')
      pltool.setAxSquare(plt.gca())
      plt.subplot(2, 2, 4)
      self.plot_corr(X, names)
      pltool.setAxSquare(plt.gca())
      plt.show()


if __name__ == '__main__':
   def main(argv):
      skip_rows = 0
      skip_cols = 0
      l_col = 0
      try:
         opts, args = getopt.getopt(argv, ['h:'], ["help",
                                                   "skip_rows=",
                                                   "skip_cols=",
                                                   "label_col="])
         for opt, arg in opts:
            if opt in ['-h', "--help"]:
               usage()
            elif opt == "skip_rows":
               skip_rows = int(arg)
            elif opt == "skip_cols":
               skip_cols = arg
            elif opt == "label_col":
               l_col = int(arg)
         filename = args[-1]
         if os.path.isfile(filename) is False:
            print 'No such file!'
            usage()
            exit(1)
         else:
            X = np.loadtxt(filename, dtype=float, delimiter=',', skiprows=skip_rows)
            y = None
            n, d = X.shape
            if l_col > d or l_col < -1:
               print 'Label column is wrongly set, default to no labels col'
               viewer = DatasetViewer()
               viewer.visualize(X[:-1], y=None)
            elif l_col == -1:
               y = X[-1]
               viewer = DatasetViewer()
               viewer.visualize(X[:-1], y)
            else:
               y = X[l_col - 1]
               viewer = DatasetViewer()
               viewer.visualize(np.c_[X[:l_col - 1], X[l_col:]], y)
      except:
         usage()
         exit(2)


   def usage():
      print '''
        python DataViz.py [options] *.csv
        [options] -s or --skip integer : skip # headlines
                  --label_col integer : which column is labels, -1 for the last row, default no labels
                  -h or --help : print usage
        '''


   main(sys.argv[1:])
