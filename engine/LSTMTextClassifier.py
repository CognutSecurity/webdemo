'''
A bidirectional LSTM sequence model used for document classification.
It is basically a sequence classification model developed in mxnet.

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import mxnet as mx
import numpy as np
import os
import pickle
import logging
import yaml
import logging.config
from BaseActor import BaseActor
from helpers.mxhelper import BasicArgparser
from BucketSeqLabelIter import BucketSeqLabelIter


class LSTMTextClassifier(BaseActor):
    """ """

    def __init__(self,
                 num_hidden=256,
                 num_embed=128,
                 input_dim=None,
                 lstm_layer=1,
                 num_classes=2,
                 params_file='',
                 checkpoint_file='',
                 save_checkpoints_path='',
                 default_bucket_key=10,
                 learning_rate=.1,
                 optimizer='sgd',
                 use_gpus=[],
                 use_cpus=[],
                 logging_root_dir='logs/',
                 logging_config='configs/logging.yaml',
                 verbose=False
                 ):

        # setup logging
        try:
            # logging_root_dir = os.sep.join(__file__.split('/')[:-1])
            logging_path = logging_root_dir + self.__class__.__name__ + '/'
            if not os.path.exists(logging_path):
                os.makedirs(logging_path)
            logging_config = yaml.safe_load(open(logging_config, 'r'))
            logging_config['handlers']['info_file_handler']['filename'] = logging_path + 'info.log'
            logging_config['handlers']['error_file_handler']['filename'] = logging_path + 'error.log'
            logging.config.dictConfig(logging_config)
        except IOError:
            logging.basicConfig(level=logging.INFO)
            logging.warning(
                "logging config file: %s does not exist." % logging_config)
        finally:
            self.logger = logging.getLogger('default')

        # setup training parameters
        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.input_dim = input_dim
        self.lstm_layer = lstm_layer
        self.num_classes = num_classes
        self.params_file = params_file
        self.checkpoint_file = checkpoint_file
        if not save_checkpoints_path:
            save_checkpoints_path = logging_root_dir + '/checkpoints/'
            if not os.path.exists(save_checkpoints_path):
                os.makedirs(save_checkpoints_path)
            self.save_checkpoints_path = save_checkpoints_path
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.ctx = []
        if use_gpus:
            self.ctx = [mx.gpu(i) for i in use_gpus]
        elif use_cpus:
            self.ctx = [mx.cpu(i) for i in use_cpus]
        else:
            self.ctx = mx.cpu(0)

    def _sym_gen(self, seq_len):
        """Dynamic symbol generator

        For variable length sequence model, we define a dynamic symbol generator
        to generate various length unrolled sequence model based on differnet cells.abs

        Args:
          seq_len(int): The sequence length to unroll

        Returns:
          mx.sym.Symbol: pred-> a symbol for the output of the sequence model

        """

        data = mx.sym.Variable(name='data')
        label = mx.sym.Variable(name='softmax_label')
        embeds = mx.symbol.Embedding(
            data=data, input_dim=self.input_dim, output_dim=self.num_embed, name='embed')
        lstm_1 = mx.rnn.LSTMCell(prefix='lstm_1_', num_hidden=self.num_hidden)
        outputs, _ = lstm_1.unroll(seq_len, inputs=embeds, layout='NTC')
        for i in range(self.lstm_layer - 1):
            new_lstm = mx.rnn.LSTMCell(
                prefix='lstm_' + str(i + 2) + '_', num_hidden=self.num_hidden)
            outputs, _ = new_lstm.unroll(seq_len, inputs=outputs, layout='NTC')
        pred = mx.sym.FullyConnected(
            data=outputs[-1], num_hidden=self.num_classes, name='logits')
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    def initialize(self, data_iter):
        """Initialize the neural network model 

        This should be called during model constructor. It tries to load NN parameters from 
        a file, if it does exist, otherwise initialize it. It sets up optimizer and optimization
        parameters as well. 

        Args:
          data_iter(mx.io.NDArrayIter): initialize the model with data iterator, it should be 
            of type BucketSeqLabelIter. 
        """

        if not isinstance(data_iter, BucketSeqLabelIter):
            err_msg = "Data iterator for this model should be of type BUcketSeqLabelIter."
            raise TypeError(err_msg)
            self.logger.error(err_msg, exc_info=True)
            return

        self.model = mx.module.BucketingModule(
            sym_gen=self._sym_gen, default_bucket_key=data_iter.default_bucket_key, context=self.ctx)
        self.model.bind(data_iter.provide_data, data_iter.provide_label)
        if self.params_file:
            try:
                self.model.load_params(self.params_file)
                self.logger.info(
                    "LSTM model parameters loaded from file: %s." % (self.params_file))
            except IOError:
                self.logger.warning(
                    "Parameters file does not exist! please check your file path.")
        else:
            self.model.init_params()
            self.logger.info("LSTM Model initialized.")

        self.model.init_optimizer(optimizer=self.optimizer,
                                  optimizer_params=(('learning_rate', self.learning_rate),))

    def step(self, data_batch):
        """Feed one data batch from data iterator to train model

        This function is called when we feed one data batch to model to update parameters. 
        it can be used in train_epochs. 
        See also: train_epochs.

        Args:
          data_batch (mx.io.DataBatch): a data batch matches the model definition
        """
        self.model.forward(data_batch=data_batch)
        metric = mx.metric.CrossEntropy()
        metric.update(data_batch.label, self.model.get_outputs())
        self.logger.debug('train step %s: %f' % (metric.get()))
        self.model.backward()
        self.model.update()

    def train_epochs(self, train_data,
                     eval_data=None,
                     num_epochs=10,
                     ):
        """Train model for many epochs with training data.

        The model will be trained in epochs and possibly evaluated with validation dataset. The
        model parameters will be saved on disk. Note that for Bucketing model, only network parameters
        will be saved in checkpoint, since model symbols need to be created according to buckets 
        which match the training data. 

        Args:
          train_data (BucketSeqLabelIter): Training data iterator  
          eval_data (BucketSeqLabelIter): Validation data iterator
          num_epochs (int): Number of epochs to train  
        """

        for e in range(num_epochs):
            train_data.reset()
            for batch in train_data:
                self.step(data_batch=batch)
            if eval_data:
                eval_data.reset()
                topk = 2
                eval_metric = mx.metric.TopKAccuracy(top_k=topk)
                self.model.score(eval_data, eval_metric)
                self.logger.info("Training epoch %d -- Evaluate top %d acccuracy: %f"
                                 % (e + 1, topk, eval_metric.get()[1]))
            saved_path = self.save_checkpoints_path + self.__class__.__name__ + '.params'
            self.model.save_params(saved_path)
            self.logger.info('Parameters saved in %s.' % (saved_path))

    def predict(self, test_data, batch_size=32):
        """Predict labels on test dataset which is a list of list of encoded tokens (integer). 

        Predict labels on a list of list of integers. As for training, test data sample is
        a list of integers mapped from token. 

        Args:
          test_data (list): A list of list of integers

        Returns:
          labels (list): a list of integers (labels)  
        """

        sample_ids = range(len(test_data))
        labels = np.zeros(shape=(len(test_data, )), dtype=int)
        scores = np.zeros(shape=(len(test_data), self.num_classes))
        tt_iter = BucketSeqLabelIter(
            test_data, sample_ids, batch_size=batch_size)
        for batch in tt_iter:
            self.model.forward(batch, is_train=False)
            out = self.model.get_outputs()[0].asnumpy()
            for logits, idx in zip(out, batch.label[0].asnumpy()):
                labels[idx] = np.argmax(logits)
                scores[idx] = logits
        return labels, scores


if __name__ == '__main__':
    '''
    Run from terminal
    '''
    # arg_parser = BasicArgparser(
    #     prog="LSTM Models with varying length inputs.").get_parser()
    # args = arg_parser.parse_args()
    # # basic parameters
    # epochs = args.epochs
    # batch_size = args.batch_size
    # lr = args.learning_rate
    # ctx = []
    # if args.gpus:
    #     for gid in args.gpus:
    #         ctx.append(mx.gpu(args.gpus[gid]))
    # elif args.cpus:
    #     for cid in args.cpus:
    #         ctx.append(mx.cpu(args.cpus[gid]))
    # else:
    #    # default
    #     ctx = mx.cpu(0)

    import numpy as np
    from termcolor import colored
    from nltk.tokenize import word_tokenize
    from sklearn.model_selection import train_test_split

    # load data
    datafile = "../datasets/npc_chat_data2.p"
    data = pickle.load(open(datafile, 'r'))
    all_sents = data['Xtr']
    sents = [word_tokenize(sent) for sent in all_sents]
    labels = np.array(data['ytr'], dtype=int) - 1
    label_names = data['label_info']
    sents_encoded, vocab = mx.rnn.encode_sentences(sents, vocab=None, invalid_key='\n',
                                                   invalid_label=-1, start_label=0)
    word_map = dict([(index, word) for word, index in vocab.iteritems()])
    print 'Total #of sentences: %d, total #of words: %d' % (len(sents_encoded), len(vocab))
    tr_data, tt_data, tr_labels, tt_labels = train_test_split(
        sents_encoded, labels, train_size=0.8)
    tr_iter = BucketSeqLabelIter(tr_data, tr_labels, batch_size=64)
    tt_iter = BucketSeqLabelIter(tt_data, tt_labels, batch_size=64)

    clf = LSTMTextClassifier(input_dim=len(vocab), num_classes=len(label_names),
                             params_file='checkpoints/LSTMTextClassifier.params')
    clf.initialize(tr_iter)
    clf.train_epochs(tr_iter, tt_iter, num_epochs=10)

    # test
    test_sents = [word_tokenize(sent) for sent in all_sents[100:400]]
    test_labels = labels[100:400]
    test_sents_encoded, _ = mx.rnn.encode_sentences(test_sents, vocab=vocab)
    preds, logits = clf.predict(test_sents_encoded, batch_size=50)
    for s, p, lgt, real in zip(all_sents[100:300], preds, logits, test_labels):
        if real == p:
            print colored(s, color='blue') + \
                colored(' -> ' + label_names[p] +
                        ' <- ' + label_names[real], color='green')
        else:
            print colored(s, color='blue') + \
                colored(' -> ' + label_names[p] +
                        ' <- ' + label_names[real], color='red')

        # print 'Logits: ', lgt
