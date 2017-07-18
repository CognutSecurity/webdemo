"""
Preprocessor is a class used to preprocess the raw data to machine learning readable dataset.
The run method in Preprocessor will iterate over each worker to process your input data, output
is a N*D numeric trainable dataset, where N is the number of training samples, and D is the feature dimensionality.

You need to define a proper preprocessor with a proper pipeline of workers for the classifier you are using.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

class Preprocessor(object):
    """
    Preprocessor main class: the pipeline for preprocessing raw data to generate vectorized feature set
    It is mainly used in the first step in any learning cycle.
    """

    def __init__(self, pipeline, feature_names=list()):
        """
        Preprocessing the raw data to build feature vectors, we use a pipeline to get our job done
        :param data_raw: raw data as in QuickTemplate.train
        :return: vectorized dataset for further process
        """

        # after initialization pipeline is a list of objects to prepare the feature vectors
        import importlib

        self._PIPELINE = list()
        self._FEATURE_NAMES = feature_names
        self._FEATURE_SIZE = 0
        for elem in pipeline:
            worker_class = getattr(importlib.import_module("mlcore.engine.PipelineWorkers"), elem['worker'])
            if elem.has_key('params'):
                worker = worker_class(elem['params'])
            else:
                worker = worker_class()
            self._PIPELINE.append(worker)


    def run(self, data_raw, restart=False):
        """
        Start processing
        :param data_raw:
        :return: feature set for training
        """

        for worker in self._PIPELINE:
            if restart or not worker.fitted:
                data_raw = worker.transform(data_raw)
            else:
                data_raw = worker.partial_transform(data_raw)

            # we set the feature names from HashParser's feature mapping
            if worker.__class__.__name__ == 'HashParser':
                self._FEATURE_NAMES = worker.feature_mapping.keys()
        self._FEATURE_SIZE = data_raw.shape[1]
        return data_raw


if __name__ == '__main__':
    import json

    #data_raw = json.load(open('../feed.json'))
    data_raw = json.load(open('../feed.json'))
    # workers = [('Tokenizer', {'language': 'english', 'nonstop': 'english'}),
    #            ('Stemmer', {'type': 'Poster'})]
    workers = [('TfidfVectorizer', {'encoding': 'utf-8'})]
    mails_in = [mail['body'] for mail in data_raw['emails_in']]
    # labels = [mail['template_']]
    # before we feed the data into preprocessor, we need to structure it as a list of something...
    prep = Preprocessor(workers)
    dataset_tr = prep.run(mails_in[:30])
    dataset_tt = prep.run(mails_in[30:])

