"""
Base class for machine learning classifier. It is an abstract class defining
methods need to be implemented in subclasses.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

from abc import ABCMeta, abstractmethod
from datetime import datetime
import dill, logging, sys
import numpy as np
import Preprocessor


class H3BaseClassifier(object):
   '''
   Abstract class for classifiers of SmartEngine.
   All smart components should inherit from this super class.
   '''

   __metaclass__ = ABCMeta

   def __init__(self,
                preprocessor=None,
                max_epoch=10,
                log_file=None,
                log_level=3):
      '''
      Initialization
      :param preprocessor: a list of preprocessors
      :param max_epoch: maximal epochs to train
      :param status: #epochs, if 0 means the classifier is never trained
      :param log_level: logging level, possible value can be
                         0: no logging
                         1: DEBUG
                         2: INFO
                         3: WARNING
                         4: ERROR
                         5: CRITICAL
      '''

      self.max_epoch = max_epoch
      self.status = 0
      self.preprocessor = preprocessor
      self.log_level = log_level
      if log_file is None:
         log_file = 'logs/' + self.__name__ + '.log'
      self.logger = self._init_logger(self.__name__, log_file, log_level)

   def _init_logger(self, name, logfile, level):
      logger = logging.getLogger(name)
      logger.setLevel(logging.DEBUG)
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      fHandler = logging.FileHandler(logfile, mode='w+', encoding='utf8')
      sHandler = logging.StreamHandler(stream=sys.stdout)
      fHandler.setFormatter(fmt=formatter)
      sHandler.setFormatter(fmt=formatter)
      sHandler.setLevel(level * 10)
      logger.addHandler(fHandler)
      logger.addHandler(sHandler)
      return logger

   @abstractmethod
   def fit(self, X, y=None):
      # train on a training dataset
      self.logger.info(self.__name__ + ' is trained on {:d} samples with {:d} features.'.format(X.shape[0], X.shape[1]))
      pass


   @abstractmethod
   def partial_fit(self, X, y=None):
      # update model on a minibatch
      self.logger.info(self.__name__ +
                       ' is updated on a mini-batch dataset with {:d} samples and {:d} features.'. \
                       format(X.shape[0], X.shape[1]))
      pass


   @abstractmethod
   def predict(self, Xtt):
      # predict outputs for test dataset
      self.logger.info(self.__name__ + ' predicts on {:d} samples.'.format(Xtt.shape[0]))
      pass

   @abstractmethod
   def decision_function(self, Xtt):
      # predict decision score on test dataset
      self.logger.info(self.__name__ + ' predicts decision scores on {:d} samples.'.format(Xtt.shape[0]))

   def save(self, path):
      # save checkpoint for the predictive model
      dill.dump(self, open(path, 'w+'))
      self.logger.info(self.__name__ + ' checkpoint is saved at {:s}.'.format(path))

   def add_preprocessor(self, pc):
      '''
      Append additional preprocessor to the list of preprocessor in this classifier.
      :param pc: an instance of preprocessor
      :return:
      '''

      if isinstance(pc, Preprocessor):
         # append a new preprocessor
         self.preprocessor.append(pc)
      else:
         self.logger.error('Invalid preprocessor! exit!')
         sys.exit('Invalid preprocessor! exit!')

   def prepare_data(self, data_blocks, restart=False):
      '''
      prepare a trainable dataset from a list data blocks each of which is processable
      by its preprocessor accordingly. Processed data blocks are concatenated as a bigger trainable dataset.
      :param data_blocks: a list of data blocks
      :return: A nxd trainable ndarray, d = sum(feature sizes of data blocks)
      '''

      begin = True
      if self.preprocessor is not None:
         nrows = 0
         if type(self.preprocessor) is not list:
            self.preprocessor = [self.preprocessor]
         if type(data_blocks) is not list:
            data_blocks = [data_blocks]
         if len(self.preprocessor) != len(data_blocks):
            self.logger.error('Num. of data blocks do not align with num. of preprocessors in classifer.')
            sys.exit()
         for pc, block in zip(self.preprocessor, data_blocks):
            if len(block) == 0:
               # empty data block
               pc._FEATURE_NAMES = []
               pc._FEATURE_SIZE = 0
               continue
            if begin:
               output = pc.run(block, restart=restart)
               nrows = output.shape[0]
               begin = False
            else:
               cur_output = pc.run(block, restart=restart)
               if cur_output.shape[0] != nrows:
                  self.logger.error('Preprocessor {:s} does not align with previous data block dimensions'.format(pc.__name__))
                  sys.exit(0)
               else:
                  output = np.c_[output, cur_output]
         return output
      else:
         self.logger.warn('No preprocessor is found in this classifier, data blocks are directly concatenated.')
         output = data_blocks[0]
         for block in data_blocks[1:]:
            output = np.c_[output, block]
         return output

   @abstractmethod
   def plot_classifier(self, **kwargs):
      '''
      Implement the plotting function in corresponding classifier class.
      '''
      pass
