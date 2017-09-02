'''
A sequence-label data iterator class. Data is a list of integer tokens, labels are
corresponding labels in integers.

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import mxnet as mx
from mxnet.io import DataIter, DataBatch, DataDesc
import numpy as np
import random, bisect


class BucketSeqLabelIter(DataIter):
   """Simple bucketing iterator for sequence classification.
   Sequence has varying length, but label is a class label.

   Parameters
   ----------
   seq : list of list of int
       Encoded seq.
   labels: list of int
   batch_size : int
       Batch size of the data.
   invalid_label : int, optional
       Key for invalid label, e.g. <end-of-sentence>. The default is -1.
   data_dtype : float, optional
       Data type of the encoding. The default data type is 'float32'.
   label_dtype : int, optional
       Data type of the label. The default data type is 'int'.
   buckets : list of int, optional
       Size of the data buckets. Automatically generated if None.
   data_name : str, optional
       Name of the data. The default name is 'data'.
   label_name : str, optional
       Name of the label. The default name is 'softmax_label'.
   """

   def __init__(self, seqs, labels,
                batch_size,
                buckets=None,
                min_bucket_key=None,
                max_bucket_key=None,
                invalid_label=-1,
                data_name='data',
                label_name='softmax_label',
                data_dtype='float32',
                label_dtype='int'):

      # if not isinstance(min_bucket_key, int) or \
      #    not isinstance(max_bucket_key, int) or \
      #    min_bucket_key <= 0 or \
      #    max_bucket_key <= 0:
      #     print "min. or max bucket key must be positive integer."
      #     sys.exit()

      super(BucketSeqLabelIter, self).__init__()
      if not buckets:
         # only consider bucket whose len >= batch_size
         buckets = [i for i, j in enumerate(np.bincount([len(s) for s in seqs]))
                    if j >= batch_size]
      buckets.sort()

      if min_bucket_key:
         buckets = [k for k in buckets if k >= min_bucket_key]
      if max_bucket_key:
         buckets = [k for k in buckets if k <= max_bucket_key]

      ndiscard = 0
      # distribute sequences in defined buckets
      self.data = [[] for _ in buckets]
      self.label = [[] for _ in buckets]
      for i, seq in enumerate(seqs):
         buck_id = bisect.bisect_left(buckets, len(seq))
         if (buck_id == 0 and len(seq) < buckets[0]) or buck_id == len(buckets):
            # sequence longer or shorter than the biggest or smallest bucket is disgarded
            ndiscard += 1
            continue
         buff = np.full((buckets[buck_id],),
                        invalid_label, dtype=data_dtype)
         buff[:len(seq)] = seq
         self.data[buck_id].append(buff)
         self.label[buck_id].append(labels[i])

      # buckets of sequences with padding
      self.data = [np.asarray(i) for i in self.data]

      print("WARNING: discarded %d sentences longer than the largest bucket." % ndiscard)

      self.batch_size = batch_size
      self.buckets = buckets
      self.data_name = data_name
      self.label_name = label_name
      self.data_dtype = data_dtype
      self.label_dtype = label_dtype
      self.invalid_label = invalid_label
      self.nddata = []
      self.ndlabel = []
      # self.major_axis = layout.find('N')
      # self.layout = layout
      self.default_bucket_key = max(buckets)

      # if self.major_axis == 0:
      self.provide_data = [DataDesc(name=self.data_name,
                                    shape=(batch_size, self.default_bucket_key),
                                    layout='NT')]
      self.provide_label = [DataDesc(name=self.label_name,
                                     shape=(batch_size,),
                                     layout='NT')]
      # elif self.major_axis == 1:
      #     self.provide_data = [DataDesc(
      #         name=self.data_name, shape=(
      #             self.default_bucket_key, batch_size),
      #         layout=self.layout)]
      #     self.provide_label = [DataDesc(
      #         name=self.label_name, shape=(
      #             self.default_bucket_key, batch_size),
      #         layout=self.layout)]
      # else:
      #     raise ValueError(
      #         "Invalid layout %s: Must by NT (batch major) or TN (time major)")
      self.idx = []
      for i, buck in enumerate(self.data):
         self.idx.extend([(i, j) for j in range(
            0, len(buck) - batch_size + 1, batch_size)])
      self.curr_idx = 0
      self.reset()

   def reset(self):
      """Resets the iterator to the beginning of the data."""
      self.curr_idx = 0
      random.shuffle(self.idx)
      # for buck in self.data:
      #     np.random.shuffle(buck)

      self.nddata = []
      self.ndlabel = []
      for buck, label in zip(self.data, self.label):
         self.nddata.append(mx.ndarray.array(buck, dtype=self.data_dtype))
         self.ndlabel.append(mx.ndarray.array(label, dtype=self.label_dtype))

   def next(self):
      """Returns the next batch of data."""
      if self.curr_idx == len(self.idx):
         raise StopIteration
      i, j = self.idx[self.curr_idx]
      self.curr_idx += 1

      # if self.major_axis == 1:
      #     data = self.nddata[i][j:j + self.batch_size].T
      #     label = self.ndlabel[i][j:j + self.batch_size].T
      # else:
      data = self.nddata[i][j:j + self.batch_size]
      label = self.ndlabel[i][j:j + self.batch_size]

      return DataBatch([data, ],
                       [label, ],
                       pad=0,
                       bucket_key=self.buckets[i],
                       provide_data=[DataDesc(
                          name=self.data_name, shape=data.shape,
                          layout='NT'), ],
                       provide_label=[DataDesc(
                          name=self.label_name, shape=label.shape,
                          layout='NT'), ])
