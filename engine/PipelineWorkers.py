"""
PipelineWorkers is a collection of workers, each of which transform a dataset input to another form.
Typical usage is that any worker accpet a dataset with n rows in a list, and outputs a n rows transformed dataset.

In machine learning, before we can pass the dataset in any learning algorithm, we should preprocess the data in
a right way. For each worker, we define a particular preprocessing step for the learning algorithm, such as tokenizer,
stemmer, tf-idf vectorizer, normalizer, standardizer and so on. You can even append or remove additional features or do
feature engineering as you wish. However, you need to make sure the order of the workers defined in your pipeline, such
that the dataset can be processed in a stream line without any conflicts.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import nltk, numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Worker(object):
    """
    Super class for all pipeline workers
    Input for every worker should be the same length of the output
    E.g., input a list of n emails returns a list of n lists of tokens
    """

    def __init__(self):
        self.fitted = False


class Tokenizer(Worker):
    """
    Word tokenizer transforms a list of documents to list of tokens,
    each element in list corresponds to a training sample.

    Tokenizer inherits nltk.word_tokenize to transform a list of texts to a list of token rows,
    each of which contains a list of tokens from the text.
    """

    def __init__(self, params={'language': 'english'}):
        super(Tokenizer, self).__init__()
        self.language = params['language']

    def transform(self, dataset):
        # print 'tokenizing...'
        tokens = list([])
        tokenizer = RegexpTokenizer(r'\w+')
        for row in dataset:
            tokens.append(tokenizer.tokenize(row.lower()))
        return tokens


class Stemmer(Worker):
    """
    Stemmer transforms a list of token rows to be stemmed tokens, e.g., moved -> move

    Stemmer inherits nltk.PorterStemmer to transform a list of token rows to a list of stemmed token rows,
    note that stemming can be problematic sometimes.
    """

    def __init__(self, params=None):
        super(Stemmer, self).__init__()

    def transform(self, dataset):
        # print 'stemming...'
        stemmer = nltk.PorterStemmer()
        stems = list([])
        for row in dataset:
            stem_row = [stemmer.stem(token) for token in row]
            stems.append(stem_row)
        return stems


class TfidfVectorizer(Worker):
    """
    TfidfVectorizer transforms a list of token rows to a list of real valued feature set

    TfidfVectorizer inherits scikit-learn's tfidf method.
    """

    def __init__(self, params):
        """
        We use the defaut tfidf vectorizer from sklearn, to support most important parameters
        :param params: 'stop_words' is a list of filtered words, default None, only support 'english'
                       'ngram_range' is a tuple (min_n, max_n) define lower and upper boundary of n-grams range
                       'max_df' is a float in [0,1] defines that a word with document frequency higher than it will be
                                ignored.
                       'min_df' same as 'max_df', word with document frequency less than it will be ignored.
                       'max_features' is a Int define the maximal number of features (most frequent) being considered
        More details see:
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        :return: instance of the TfidfVectorizer worker
        """

        if params.has_key('stop_words'):
            _stop_words = params['stop_words']
        else:
            _stop_words = None
        if params.has_key('ngram_range'):
            _ngram_range = params['ngram_range']
        else:
            _ngram_range = (1, 2)
        if params.has_key('max_df'):
            _max_df = params['max_df']
        else:
            _max_df = 1.0
        if params.has_key('min_df'):
            _min_df = params['min_df']
        else:
            _min_df = 1
        if params.has_key('max_features'):
            _max_features = params['max_features']
        else:
            _max_features = None

        super(TfidfVectorizer, self).__init__()
        self._worker = TfidfVec(stop_words=_stop_words,
                                ngram_range=_ngram_range,
                                max_df=_max_df,
                                min_df=_min_df,
                                max_features=_max_features)

    def transform(self, dataset):
        """
        transform is called to initialize the vectorizer, status will be refreshed
        :param dataset: input dataset
        :return: output vectorized Tfidf dataset in ndarray
        """

        # print 'tf-idf vectorizing...'
        samples = self._worker.fit_transform(dataset)
        self.fitted = True
        return samples.toarray()

    def partial_transform(self, dataset):
        """
        partial transform uses current vectorizer to fit more data, typically used for tranform testing data
        :param dataset: input dataset
        :return: output vectorized Tfidf dataset in ndarray
        """
        if self.fitted is False:
            print 'TfidfVectorizer is not yet initialized on any dataset, exit...\n'
            return False
        else:
            # use existing vectorizer
            # print 'tf-idf vectorizing...'
            samples = self._worker.transform(dataset)
            return samples.toarray()


class Normalizer(Worker):
    """
    Normalize input samples to unit norm. See also:
        - sklearn.preprocessing.normalize
    """

    def __init__(self, params={'norm': 'l2'}):
        """
        Initialisation
        :param params: params['norm'] can be 'l1' or 'l2'
        :return: instance of Normalizer
        """
        super(Normalizer, self).__init__()
        if not params.has_key('norm') or params['norm'] not in ['l1', 'l2']:
            print 'Error initialze a normalizer, needs set as l1 or l2'
        self.norm = params['norm']

    def transform(self, dataset):
        """
        transform is called to initialize the vectorizer, status will be refreshed
        :param dataset: input dataset is a ndarray with shape=(n,d), where n is the sample size, and d is the feature size
        :return: normalized dataset
        """

        # print 'normalizing...'
        samples = normalize(dataset, norm=self.norm, axis=1)
        return samples


class FeatureScaler(Worker):
    """
    Scale an input dataset to zero-mean and unit variance.
    It supports now standard scaling and minmax scaling, see also:
        - sklearn.preprocessing.StandardScaler
        - sklearn.preprocessing.MinMaxScaler
    """

    def __init__(self, params=None):
        """
        Initialization
        :param params: params['online'] Boolean value indicates if fit the data online
        :return: instance of StandardScaler
        """
        super(FeatureScaler, self).__init__()
        if params.has_key('type'):
            _type = params['type']
        else:
            _type = 'standard'
        if _type == 'standard':
            self._worker = StandardScaler()
        if _type == 'minmax':
            self._worker = MinMaxScaler()

    def transform(self, dataset):
        """
        transform is called to scale the dataset feature to zero-mean and unit variance.
        :param dataset: Nxd ndarray with shape=(n,d), where n is the sample size, and d is the feature size
        :return: scaled dataset
        """

        dataset = numpy.array(dataset)
        if dataset.shape[1] == 0:
            # if the input contains nothing...
            return numpy.array([])
        samples = self._worker.fit_transform(dataset)
        self.fitted = True
        return samples

    def partial_transform(self, dataset):
        """
        partial tranform use current scaler to scale input samples to zero-mean and unit variance.
        :param dataset: Nxd ndarray with shape=(m,d)
        :return: scaled samples
        """

        if not self.fitted:
            print 'FeatureScaler is not yet initialized on any dataset, exit...\n'
            return False
        else:
            # use existing scaler
            dataset = numpy.array(dataset)
            if dataset.shape[1] == 0:
                # if the input contains nothing...
                return numpy.array([])
            samples = self._worker.transform(dataset)
            return samples


class VaderSentiment(Worker):
    """
    VaderSentiment transforms a list of text corpus to a list of sentiment intensity scores.
    We use the Vader sentiment analysis tools provided by NLTK, this worker is stateless.
    [Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
     Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.]
    """

    def __init__(self, params=None):
        """
        Initializer
        :param params: params['corpuses'] is the only params
        :return:
        """
        super(VaderSentiment, self).__init__()

    def transform(self, dataset):
        """
        append an additional sentiment scores col. in the end of the dataset
        :param dataset: a list of N text corpuses
        :return: Nx1 vector of sentiment scores
        """

        sentiment_scores = list()
        sid = SentimentIntensityAnalyzer()
        for row in dataset:
            # tokenzie sentences
            lines_list = nltk.tokenize.sent_tokenize(row)
            # we compute the compouond scores for each sentence in one single text
            scores = [sid.polarity_scores(line)['compound'] for line in lines_list]
            # we compute avg. sentiment score for the text
            sentiment_scores.append(numpy.mean(scores))
        sentiment_scores = numpy.array(sentiment_scores).reshape(len(sentiment_scores), 1)
        return sentiment_scores


class HashParser(Worker):
    """
    HashParser parses a list of hashes to get a list of features
    """

    def __init__(self, params=None):
        """
        Initializer
        :param params: params['corpuses'] is the only params
        :return:
        """
        super(HashParser, self).__init__()
        # keys contain a list of keys found in hash
        self.feature_mapping = OrderedDict()

    def transform(self, dataset):
        """
        convert a list of hashes to a list of features
        :param dataset: a list of hashes
        :return: a list of feature vectors
        """

        # find all keys/values first
        for item in dataset:
            for k in item.keys():
                if k not in self.feature_mapping.keys():
                    value_list = list()
                    value_list.append(item[k])
                    self.feature_mapping[k] = value_list
                else:
                    if item[k] not in self.feature_mapping[k]:
                        self.feature_mapping[k].append(item[k])

        features = []
        for item in dataset:
            row = []
            for k in self.feature_mapping.keys():
                if item.has_key(k):
                    try:
                        idx = self.feature_mapping[k].index(item[k])
                        row.append(idx + 1.0)
                    except ValueError:
                        row.append(-1.0)
                else:
                    row.append(0.0)
            features.append(row)

        self.fitted = True
        return numpy.array(features)

    def partial_transform(self, dataset):
        """
        partial_transform convert a list of dicts to features using current available keys
        :param dataset: a list of dict
        :return: a list of features
        """

        if not self.fitted:
            print 'HashParser is not yet initialized on any dataset, exit...\n'
            return False
        else:
            features = []
            for item in dataset:
                row = []
                for k in self.feature_mapping.keys():
                    if item.has_key(k):
                        try:
                            idx = self.feature_mapping[k].index(item[k])
                            row.append(idx + 1)
                        except ValueError:
                            row.append(-1)
                    else:
                        row.append(0)
                features.append(row)
            return numpy.array(features)


