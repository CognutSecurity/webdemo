#!/usr/bin/env python
# coding: utf-8

'''
Main routine for gaussian copula Bayesnet

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import sys, argparse, logging
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy import linalg
from sklearn.covariance import graph_lasso, EmpiricalCovariance, ledoit_wolf, shrunk_covariance
from sklearn.preprocessing import StandardScaler
from mlcore.utils.bn_helper import *
from mlcore.utils.dataset_helper import read_csv
from libpgm import pgmlearner
from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest


class H3BayesNet(object):
   '''
   class for Gaussian copula bayes network
   '''

   def __init__(self,
                vdata=None,  # a JSON object contains nodes information, see libpgm's input format
                vnames=list([]),  # a list of names for vertexes
                edges=list([]),  # a list of tuples (parent, child) to define graph edges
                method='glasso',  # method used to estimate precision matrix
                marginal='kde',  # method used to evaluate marginal densities on X
                penalty=1.0,  # penalty coefficient to evaluate precision matrix
                alpha=0.05,  # threshold for conditional independence
                eps=1e-4,  # tolerance of convergence
                theta=1e-3,  # truncate density threshold
                max_iter=1000,
                verbose=False,
                normalize=True,
                bin=5,
                pval=0.05):  # maximal iterations for optimization

      self.Vdata = vdata
      self.vertexes = vnames
      self.edges = edges
      self.method = method
      self.marginal = marginal
      self.penalty = penalty
      self.alpha = alpha
      self.eps = eps
      self.max_iter = max_iter
      self.theta = theta
      if normalize:
         self.scaler = StandardScaler()
      else:
         self.scaler = None
      self.kernels = []
      self.est_cov = None
      self.precision_ = None
      self.corr = None
      self.conditional_independences_ = None
      self.ci_coef_ = None
      self.dag_raw = None
      self.dag_moral = None
      self.verbose = verbose
      self.bins = bin
      self.pval = pval

   def get_params(self, key=None):
      if key is None:
         return self.__dict__

      elif key in self.__dict__:
         return self.__dict__[key]

      else:
         print colored('No such attribut in class ' + self.__class__.__name__, color='yellow')
         return None

   def set_params(self, params=None):
      if params is not None:
         for key in params.keys():
            if key in self.__dict__:
               self.__dict__[key] = params[key]
            else:
               print colored('Warning: {:s} is not attribute of class {:s}.'.format(key, self.__class__.__name__),
                             color='yellow')
               pass

   def fit(self, X):
      '''
      Copulafit using Gaussian copula with marginals evaluated by Gaussian KDE
      Precision matrix is evaluated using specified method, default to graphical LASSO
      :param X: input dataset
      :return: estimated precision matrix rho
      '''

      N, d = X.shape
      if self.scaler is not None:
         X_scale = self.scaler.fit_transform(X)
      else:
         X_scale = X
      if len(self.vertexes) == 0:
         self.vertexes = [str(id) for id in range(d)]

      self.theta = 1.0 / N
      cum_marginals = np.zeros_like(X)
      inv_norm_cdf = np.zeros_like(X)
      # inv_norm_cdf_scaled = np.zeros_like(X)
      self.kernels = list([])
      # TODO: complexity O(Nd) is high
      if self.verbose:
         colored('>> Computing marginals', color='blue')
      for j in range(cum_marginals.shape[1]):
         self.kernels.append(gaussian_kde(X_scale[:, j]))
         cum_pdf_overall = self.kernels[-1].integrate_box_1d(X_scale[:, j].min(), X_scale[:, j].max())
         for i in range(cum_marginals.shape[0]):
            cum_marginals[i, j] = self.kernels[-1].integrate_box_1d(X_scale[:, j].min(),
                                                                    X_scale[i, j]) / cum_pdf_overall
            # truncate cumulative marginals
            if cum_marginals[i, j] < self.theta:
               cum_marginals[i, j] = self.theta
            elif cum_marginals[i, j] > 1 - self.theta:
               cum_marginals[i, j] = 1 - self.theta
            # inverse of normal CDF: \Phi(F_j(x))^{-1}
            inv_norm_cdf[i, j] = norm.ppf(cum_marginals[i, j])
            # scaled to preserve mean and variance: u_j + \sigma_j*\Phi(F_j(x))^{-1}
            # inv_norm_cdf_scaled[i, j] = X_scale[:, j].mean() + X_scale[:, j].std() * inv_norm_cdf[i, j]

      if self.method == 'mle':
         # maximum-likelihood estiamtor
         empirical_cov = EmpiricalCovariance()
         empirical_cov.fit(inv_norm_cdf)
         if self.verbose:
            print colored('>> Running MLE to estiamte precision matrix', color='blue')

         self.est_cov = empirical_cov.covariance_
         self.corr = scale_matrix(self.est_cov)
         self.precision_ = inv(empirical_cov.covariance_)

      if self.method == 'glasso':
         if self.verbose:
            print colored('>> Running glasso to estiamte precision matrix', color='blue')

         empirical_cov = EmpiricalCovariance()
         empirical_cov.fit(inv_norm_cdf)
         # shrunk convariance to avoid numerical instability
         shrunk_cov = shrunk_covariance(empirical_cov.covariance_, shrinkage=0.8)
         self.est_cov, self.precision_ = graph_lasso(emp_cov=shrunk_cov, alpha=self.penalty,
                                                     verbose=self.verbose,
                                                     max_iter=self.max_iter)
         self.corr = scale_matrix(self.est_cov)

      if self.method == 'ledoit_wolf':
         if self.verbose:
            print colored('>> Running ledoit_wolf to estiamte precision matrix', color='blue')

         self.est_cov, _ = ledoit_wolf(inv_norm_cdf)
         self.corr = scale_matrix(self.est_cov)
         self.precision_ = linalg.inv(self.est_cov)


      if self.method == 'pc':
         clf = pgmlearner.PGMLearner()
         data_list = list([])
         for row_id in range(X_scale.shape[0]):
            instance = dict()
            for i, n in enumerate(self.vertexes):
               instance[n] = X_scale[row_id, i]
            data_list.append(instance)
         graph = clf.lg_constraint_estimatestruct(data=data_list,
                                                  pvalparam=self.pval,
                                                  bins=self.bins)
         dag = np.zeros(shape=(len(graph.V), len(graph.V)))
         for e in graph.E:
            dag[self.vertexes.index(e[0]), self.vertexes.index(e[1])] = 1
         self.conditional_independences_ = dag

      if self.method == 'ic':
         df = dict()
         variable_types = dict()
         for j in range(X_scale.shape[1]):
            df[self.vertexes[j]] = X_scale[:, j]
            variable_types[self.vertexes[j]] = 'c'
         data = pd.DataFrame(df)
         # run the search
         ic_algorithm = IC(RobustRegressionTest, data, variable_types, alpha=self.pval)
         graph = ic_algorithm.search()
         dag = np.zeros(shape=(X_scale.shape[1], X_scale.shape[1]))
         for e in graph.edges(data=True):
            i = self.vertexes.index(e[0])
            j = self.vertexes.index(e[1])
            dag[i, j] = 1
            dag[j, i] = 1
            arrows = set(e[2]['arrows'])
            head_len = len(arrows)
            if head_len > 0:
               head = arrows.pop()
               if head_len == 1 and head == e[0]:
                  dag[i, j] = 0
               if head_len == 1 and head == e[1]:
                  dag[j, i] = 0
         self.conditional_independences_ = dag

      # finally we fit the structure
      self.fit_structure(self.precision_)

   def fit_structure(self, precision):
      '''
      from estimated precision matrix to completed DAG
      :return:
      '''

      if self.method == 'pc' or self.method == 'ic':
         self.Vdata = adj2json(self.conditional_independences_, node_names=self.vertexes)
         return

      partial_inv_corr = scale_matrix(precision)

      if self.verbose:
         print colored('>> Estimated covariance matrix', color='blue')
         print '\t' + str(self.est_cov)
         print colored('>> Estimated correlation matrix', color='blue')
         print '\t' + str(self.corr)
         print colored('>> Estimated precision matrix', color='blue')
         print '\t' + str(self.precision_)
         print colored('>> Partial inverse correlation matrix', color='blue')
         print '\t' + str(partial_inv_corr)

      self.dag_raw = to_adjacent_matrix(partial_inv_corr, threshold=self.alpha)
      partial_dag, colliders = self.detriangulation(self.dag_raw, self.corr, self.alpha)
      self.dag_moral = partial_dag.copy()
      if self.verbose:
         print colored('>> Runninig detriangulation, the partial directed graph:', color='blue')
         print partial_dag
         print colored('>> Detected colliders:', color='blue')
         print colliders

      partial_dag = self.propogate_constraints(partial_dag, unknown_edges=get_unknown_edges(partial_dag))
      if self.verbose:
         print colored('>> Propagate constraints, the maximal directed graph:', color='blue')
         print partial_dag

      self.conditional_independences_ = partial_dag
      self.ci_coef_ = np.multiply(self.conditional_independences_, partial_inv_corr)
      self.Vdata = adj2json(self.conditional_independences_, node_names=self.vertexes)

   def score_function(self, X):

      '''
      compute log-likelihood on samples X
      :param X: mxd ndarray
      :return: a list log-probability on each sample
      '''

      if self.method == 'pc':
         print colored('LLE for PC algorithm not supported.', color='red')
         return -1

      cum_marginals = np.zeros_like(X)
      inv_norm_cdf = np.zeros_like(X)

      # transform X
      X_scale = self.scaler.transform(X)
      for j, k in enumerate(self.kernels):
         cum_pdf_overall = k.integrate_box_1d(X_scale[:, j].min(), X_scale[:, j].max())
         for i in range(cum_marginals.shape[0]):
            cum_marginals[i, j] = k.integrate_box_1d(X_scale[:, j].min(), X_scale[i, j]) / cum_pdf_overall
            # truncate cumulative marginals
            if cum_marginals[i, j] < self.theta:
               cum_marginals[i, j] = self.theta
            elif cum_marginals[i, j] > 1 - self.theta:
               cum_marginals[i, j] = 1 - self.theta
            # inverse of normal CDF: \Phi(F_j(x))^{-1} f(x) in nonparanomal paper
            inv_norm_cdf[i, j] = norm.ppf(cum_marginals[i, j])

      lle_unnormalized = np.log(np.linalg.det(self.precision_)) - \
                         np.trace(inv_norm_cdf.dot(self.precision_).dot(inv_norm_cdf.T))
      # lle_normalized = lle_unnormalized - np.log(np.exp(lle_unnormalized).sum())
      lle_per_instance = lle_unnormalized / X.shape[0]
      return lle_per_instance

   def propogate_constraints(self, partial_dag, unknown_edges):
      '''
      Propagate constaints to obtain a maximal directed acyclic graph
      The order of rules is very critical. Acyclicity should be always firstly
      considered unless there might be a cycle in final result!
      :param partial_dag: partially directed DAG from detriangualtion
      :param unknown_edges: a list edges whose direction are not yet determined
      :return: ndarray of maximal DAG
      '''

      changed = 1
      directable = 0
      # print colored('>> Propagating constraints... ', color='green')
      while changed:
         # if graph is changed do not stop propagating constraints
         changed, directable = 0, 0

         # No cycles
         for id, edge in enumerate(unknown_edges):
            p, q = edge[0], edge[1]
            if find_path(partial_dag, p, q):
               partial_dag[q, p] = 0
               directable = 1
               del unknown_edges[id]
               break
            elif find_path(partial_dag, q, p):
               partial_dag[p, q] = 0
               directable = 1
               del unknown_edges[id]
               break
         if directable:
            if self.verbose:
               print colored('Cycle found for {:s}-{:s}'.format(self.vertexes[p], self.vertexes[q]), color='green')
            changed = 1
            continue

         # No new V-node
         for id, edge in enumerate(unknown_edges):
            p, q = edge[0], edge[1]
            _, p_incomings, _ = get_neighbors(partial_dag, p)
            if len(p_incomings) > 0:
               for income in p_incomings:
                  if partial_dag[income, q] == 0 and partial_dag[q, income] == 0:
                     partial_dag[q, p] = 0
                     directable = 1
                     del unknown_edges[id]
                     break
               if directable: break
            else:
               _, q_incomings, _ = get_neighbors(partial_dag, q)
               for income in q_incomings:
                  if partial_dag[income, p] == 0 and partial_dag[p, income] == 0:
                     partial_dag[p, q] = 0
                     directable = 1
                     del unknown_edges[id]
                     break
               if directable: break
         if directable:
            if self.verbose:
               print colored('V-node found for {:s}-{:s}'.format(self.vertexes[p], self.vertexes[q]), color='green')
            changed = 1
            continue

         # No diamond structure
         for id, edge in enumerate(unknown_edges):
            p, q = edge[0], edge[1]
            if is_diamond(partial_dag, p, q):
               partial_dag[q, p] = 0
               directable = 1
               del unknown_edges[id]
               break
            elif is_diamond(partial_dag, q, p):
               partial_dag[p, q] = 0
               directable = 1
               del unknown_edges[id]
               break
         if directable:
            if self.verbose:
               print colored('Diamond found for {:s}-{:s}'.format(self.vertexes[p], self.vertexes[q]), color='green')
            changed = 1
            continue

      return partial_dag

   def detriangulation(self, graph, mCorr, alpha):
      '''
      find out hanged triangulars from given nodes
      :param: graph is a list
      :param: alpha is a thredshold
      :return: dag and collider 
      '''

      # print colored('>> detrangulating... ', color='yellow')

      # make sure graph is a numpy array
      dag = np.array(graph)
      # make sure: diagonal is all zeros
      dag[np.diag_indices_from(dag)] = 0.

      node_size = dag.shape[0]
      collider_list = []
      for p in xrange(node_size):

         for q in xrange(p + 1, node_size):

            # only iterate unknown edges, not on p==q
            if dag[p, q] != 0 and dag[q, p] != 0:

               flag, colliders = is_marriged_edge(dag, mCorr, p, q, alpha, verbose=self.verbose)
               # flag = -1
               # colliders = []
               for c in colliders:
                  if find_path(dag, c, p) or find_path(dag, c, q):
                     flag = -1
                     break
               if flag != -1:
                  dag = set_marridged_edge_free(dag, p, q, colliders)
                  for c in colliders:
                     # if contains(colliders[k] in colliderList) == False:
                     if c not in collider_list:
                        # in matlab is [colliderList, colliders(k)]
                        # collider_list = np.concatenate(collider_list, c, axis=0)
                        collider_list.append(c)
      return dag, collider_list


if __name__ == '__main__':

   # parse argumenents
   argparser = argparse.ArgumentParser(prog='gcdag',
                                       usage="this program learns a directed acyclic graph from a dataset.",
                                       description='''
                                          We use gaussian copula and L1-regularization
                                          to estimate the inverse covariance matrix (precision matrix), then construct the
                                          equivalent DAG class from the precision matrix.
                                       ''',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   argparser.add_argument("--train", help="File path to training dataset.")
   argparser.add_argument("--log_level", default='INFO',
                          help="Set the log level in console, you can set [DEBUG, ERROR, INFO, WARNING]")
   argparser.add_argument("--log_path", help="File path to save logs", default="logs/untitled.log")
   argparser.add_argument("--alpha", help="threshold for significant level, default 0.1", default=0.1, type=float)
   argparser.add_argument("--bins", help="Bins for discretization, default 5", default=5, type=int)
   argparser.add_argument("--pval", help="P-value significance level, default 0.05", default=0.05, type=float)
   argparser.add_argument("--penalty", help="penalty for gLASSO, default 0.01", default=0.01, type=float)
   argparser.add_argument("--method", help="which method to run.. default ledoit_wolf", default='ledoit_wolf')

   args = argparser.parse_args()
   log_path = args.log_path
   if str.lower(args.log_level) == 'debug':
      console_log_level = logging.DEBUG
   elif str.lower(args.log_level) == 'info':
      console_log_level = logging.INFO
   elif str.lower(args.log_level) == 'warning':
      console_log_level = logging.WARNING
   elif str.lower(args.log_level) == 'error':
      console_log_level = logging.ERROR
   else:
      print colored('Log level is not set appropriately, see help by --help.', color='yellow')
      console_log_level = logging.DEBUG


   # initialize logger
   def init_logging(log_path, level=logging.DEBUG):
      """
      Initialize the logging functionalities
      :return: Logger
      """
      logging.basicConfig(format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
      logger = logging.getLogger('gcdag.log')
      logger.setLevel(level)
      return logger

   logger = init_logging(log_path, level=console_log_level)
   input_data, node_names = read_csv(args.train)
   clf = H3BayesNet(alpha=args.alpha, method=args.method, vnames=node_names, pval=args.pval, bin=args.bins,
                    penalty=args.penalty, verbose=True)
   clf.fit(input_data)
   print clf.ci_coef_
   print clf.conditional_independences_
