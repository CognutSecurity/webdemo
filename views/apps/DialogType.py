import os, re, nltk, pickle, dill, json, cherrypy
import numpy as np, jinja2 as jj2
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import TreebankWordTokenizer, RegexpTokenizer
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, LabelSet

env = jj2.Environment(loader=jj2.FileSystemLoader('./templates/DialogType'))

class Mails(object):
   def __init__(self):
      # NEED TO RESET PATH IF WEB RUNS ALONE
      self.TRAIN_MODEL_PATH = cherrypy.config['checkpoints_path']
      self.DATAPATH = cherrypy.config['data_path']
      # Where we save user uploaded JSON file
      self.UPLOADED_DIR = cherrypy.config['uploads_path']
      # MUST COMPLY A CERTAIN JSON FORMAT, DEFAULT WITH A SIMPLE CAMPAGIN
      # FILE
      self.CAMPAIGN_FILE = ''.join([self.UPLOADED_DIR, 'feed.json'])
      # Pretrained Glove word embedding JSON dataset used as Lookup table
      # See more: http://nlp.stanford.edu/projects/glove/
      self.GLOVE_SET = ''.join([self.DATAPATH, 'glove.twitter.27B.25d.json'])
      self.CAMPAIGN_VECTORS = ''.join([self.DATAPATH, self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.vectors.json'])
      self.CAMPAIGN_KNN = ''.join([self.DATAPATH, self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.knn.json'])

   def softmax(self, x):
      """Compute softmax values for each sets of scores in x."""
      sf = np.exp(x)
      sf = sf / np.sum(sf, axis=0)
      return sf

   def retrainModel(self):

      # build classifier for act type on NPS chat dataset
      # see: http://faculty.nps.edu/cmartell/NPSChat.htm
      # CHANGE THE CAMPAIGN FILE HERE!!! PUT THE FILE IN THE './datasets/'
      # FOLDER

      TRAINDATASET_FILE = ''.join([self.DATAPATH, 'npc_chat_data2.p'])
      nps_chat_data = pickle.load(open(TRAINDATASET_FILE, 'rb'))
      label_names = nps_chat_data['label_info']
      use_labels = [1, 5, 6, 7, 9, 10, 14, 15]
      alpha = 1e-4
      # npc chat dataset
      train_idx = []
      tfidfVec = TfidfVectorizer(binary=True,
                                 ngram_range=(1, 2),
                                 tokenizer=TreebankWordTokenizer().tokenize,
                                 strip_accents='ascii',
                                 min_df=1e-4,
                                 max_df=.9,
                                 max_features=1500)
      npc_Xtr = tfidfVec.fit_transform(nps_chat_data['Xtr'])
      npc_ytr = np.array(nps_chat_data['ytr'])
      for idx in use_labels:
         train_idx.extend(np.where(npc_ytr == idx)[0].tolist())
      # choose dataset with particular types that we are interested
      npc_Xtr = npc_Xtr[train_idx, :]
      npc_ytr = npc_ytr[train_idx]
      # train a stochastic logistic regression model using L2 norm
      clf = SGDClassifier(loss='log', alpha=alpha, penalty='l2', warm_start=True, verbose=False,
                          class_weight='balanced')
      clf.fit(npc_Xtr, npc_ytr)

      # save pretrained model
      DlgActTypeClfModel = dict()
      DlgActTypeClfModel['clf'] = clf  # Predictive model after training
      # Word vectorizer after trainining
      DlgActTypeClfModel['vectorizer'] = tfidfVec
      DlgActTypeClfModel['label_names'] = label_names  # Types names list
      DlgActTypeClfModel['user_labels'] = use_labels  # Particular types Ids
      fw = open(''.join([self.TRAIN_MODEL_PATH, 'DlgActTypeClf.p']), 'w+')
      dill.dump(DlgActTypeClfModel, fw)
      fw.close()
      cherrypy.log("[INFO] New model is trained and saved in {} ... OK".format(
         ''.join([self.TRAIN_MODEL_PATH, 'DlgActTypeClf.p'])))
      return DlgActTypeClfModel

   def populateMails(self):
      '''
      Using current server campaign file to populate main view
      '''

      # TODO: if file does not exist
      dataset = json.load(open(self.CAMPAIGN_FILE))

      # load model or retrain
      model_path = ''.join([self.TRAIN_MODEL_PATH, 'DlgActTypeClf.p'])
      if os.path.isfile(model_path):
         # load from file
         actTypeModel = dill.load(open(model_path))
      else:
         # retrain it
         actTypeModel = self.retrainModel()
      clf = actTypeModel['clf']
      tfidfVec = actTypeModel['vectorizer']
      label_names = actTypeModel['label_names']
      use_labels = actTypeModel['user_labels']

      # predict labels
      mail_list = list()
      for mail in dataset['emails_in']:
         mymail = dict()
         sent_list = list()
         if len(mail['body']) > 0:
            mymail['mid'] = mail['email_id']
            # only add a mail to results when the body is not null
            # convert mail body to a HTML paragraph for tagging purpose
            reg_obj = re.compile(r'([a-zA-Z\-]+)')
            sents = nltk.tokenize.sent_tokenize(mail['body'])
            sents_tagged = []
            sent_id = 0
            for sent in sents:
               sents_tagged.append("<span id='{}-{}'>".format(mymail['mid'], str(sent_id)) +
                                   reg_obj.sub(r'<span class="word_tag">\1</span>', sent) +
                                   "</span>")
               sent_id = sent_id + 1
            sents_tagged = [s.replace('\n', '<br>') for s in sents_tagged]
            mymail['message'] = "".join(sents_tagged)

            # compute sentences type probabilities
            sents_vec = tfidfVec.transform(sents)
            sents_labels = clf.predict(sents_vec)
            # TODO:  confirm class label order
            lin_var = clf.coef_.dot(sents_vec.T.todense()) + \
                      np.repeat(clf.intercept_.reshape(
                         clf.intercept_.size, 1), sents_labels.size, axis=1)
            for j in range(0, lin_var.shape[1]):
               sent = dict()
               sent['content'] = sents[j]
               sent['probs'] = np.array(
                  self.softmax(lin_var[:, j])).ravel()
               prob_desc_sorted = sent['probs'].argsort()[::-1]
               sent['probs'] = ['{:.2f}'.format(
                  sent['probs'][j]) for j in prob_desc_sorted]
               sent['labels'] = [label_names[use_labels[j] - 1]
                                 for j in prob_desc_sorted]
               sent['idlist'] = range(len(sent['labels']))
               sent_list.append(sent)
            mymail['sents'] = sent_list
            mail_list.append(mymail)
      # build word embedding
      # script, div = self.buildWordEmbedding(True)
      if os.path.exists(self.CAMPAIGN_VECTORS) is False:
         self.dumpCampaignVectors()
      return mail_list

   def dumpCampaignVectors(self):
      campagin_vectors = dict()
      # return similar words based on current embedding, only support one word
      feedfile = self.CAMPAIGN_FILE
      corpus = []
      # read tokens from CAMPAIGN_FILE
      tokenizer = TreebankWordTokenizer()
      allmails = json.load(open(feedfile, 'rb'))
      for mail in allmails['emails_in']:
         tokens = tokenizer.tokenize(mail['body'])
         corpus.extend([t.lower() for t in tokens])
      pretrained_vectors = json.load(open(self.GLOVE_SET, 'r'))
      for token in set(corpus):
         if pretrained_vectors.has_key(token):
            campagin_vectors[token] = pretrained_vectors[token]
      fw = open(self.CAMPAIGN_VECTORS, 'w+')
      json.dump(campagin_vectors, fw)
      fw.close()
      cherrypy.log("[INFO] Campagin word vectors are dumped in: {} ... OK".format(self.CAMPAIGN_VECTORS))

   def dumpWordDists(self):
      # compute word distances pairwise and store it in local file
      if os.path.exists(self.CAMPAIGN_VECTORS) is False:
         self.dumpCampaignVectors()
      pretrained_vectors = json.load(open(self.CAMPAIGN_VECTORS, 'rb'))
      labels = []
      D = []
      for token in pretrained_vectors.keys():
         labels.append(token)
         D.append(pretrained_vectors[token].strip().split(' '))
      D = np.array(D, dtype="float")
      dists = squareform(pdist(D, 'euclidean'))
      knn = dict()
      for id, l in enumerate(labels):
         knn[l] = [labels[j] for j in np.argsort(dists[id, :])]
      fw = open(self.CAMPAIGN_KNN, 'w+')
      json.dump(knn, fw)
      fw.close()
      cherrypy.log("[INFO] Campagin word distances are dumped in: {} ... OK".format(self.CAMPAIGN_KNN))

   def buildWordEmbedding(self, pretrained=False):
      '''
      Build word embedding according to current text corpus or pretrained model
      Return: script, div for embedded bokeh plotting
      '''

      D = []
      labels = []
      corpus = []
      GLOVE_PATH = "./glove/"
      feedfile = self.CAMPAIGN_FILE
      # corpus path
      CORPUS = ''.join([
         GLOVE_PATH, 'corpus_', self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.txt'])
      SAVE_FILE = ''.join([
         GLOVE_PATH, 'vectors_', self.CAMPAIGN_FILE.split('/')[-1].split('.')[0]])

      # read tokens from CAMPAIGN_FILE
      tokenizer = TreebankWordTokenizer()
      allmails = json.load(open(feedfile, 'rb'))
      for mail in allmails['emails_in']:
         tokens = tokenizer.tokenize(mail['body'])
         corpus.extend(tokens)

      # if not pretrained and the corresponding embedding file does not exist
      if pretrained is False and os.path.exists(SAVE_FILE + '.txt') is False:
         # intermediate file paths
         VOCAB_FILE = ''.join([
            GLOVE_PATH, 'vocab_', self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.txt'])
         COOCCURRENCE_FILE = ''.join([
            GLOVE_PATH, 'coocurance_', self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.bin'])
         COOCCURRENCE_SHUF_FILE = ''.join([
            GLOVE_PATH, 'coocurance_shuf_', self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.bin'])

         # training parameters
         VERBOSE = 2
         MEMORY = 4.0
         VOCAB_MIN_COUNT = 5
         VECTOR_SIZE = 50
         MAX_ITER = 15
         WINDOW_SIZE = 15
         BINARY = 2
         NUM_THREADS = 8
         X_MAX = 10

         # write campagin tokens to file
         fd = open(CORPUS, 'w+')
         fd.write(' '.join(corpus).encode('utf8'))
         fd.close()

         # 1. run unigram vocab builder
         os.system("{}build/vocab_count -min-count {} -verbose {} < {} > {}".format(GLOVE_PATH,
                                                                                    VOCAB_MIN_COUNT, VERBOSE, CORPUS,
                                                                                    VOCAB_FILE))
         # 2. words coocurance
         os.system(
            "{}build/cooccur -memory {} -vocab-file {} -verbose {} -window-size {} < {} > {}".format(GLOVE_PATH, MEMORY,
                                                                                                     VOCAB_FILE,
                                                                                                     VERBOSE,
                                                                                                     WINDOW_SIZE,
                                                                                                     CORPUS,
                                                                                                     COOCCURRENCE_FILE))
         # 3. shuffle
         os.system("{}build/shuffle -memory {} -verbose {} < {} > {}".format(GLOVE_PATH,
                                                                             MEMORY, VERBOSE, COOCCURRENCE_FILE,
                                                                             COOCCURRENCE_SHUF_FILE))
         # 4. glove model
         os.system(
            "{}build/glove -save-file {} -threads {} -input-file {} -x-max {} -iter {} -vector-size {} -binary {} -vocab-file {} -verbose {}".format(
               GLOVE_PATH, SAVE_FILE, NUM_THREADS, COOCCURRENCE_SHUF_FILE, X_MAX, MAX_ITER, VECTOR_SIZE, BINARY,
               VOCAB_FILE, VERBOSE))

         fd = open(SAVE_FILE + '.txt', 'rb')
         for row in fd:
            row = row.split(' ')
            labels.append(row[0])
            D.append(row[1:])
         fd.close()
      elif pretrained:
         # use pretrained Glove vectors
         if os.path.exists(self.CAMPAIGN_VECTORS) is False:
            self.dumpCampaignVectors()
         pretrained_vectors = json.load(open(self.CAMPAIGN_VECTORS, 'rb'))
         for token in pretrained_vectors.keys():
            labels.append(token)
            D.append(pretrained_vectors[token].strip().split(' '))
      else:
         # SAVE_FILE exists, open it directly
         # Built from existing Glove Model
         fd = open(SAVE_FILE + '.txt', 'rb')
         for row in fd:
            row = row.split(' ')
            labels.append(row[0])
            D.append(row[1:])
         fd.close()

      # plot words embedding
      p = figure(title="Word embedding plots", webgl=True,
                 plot_width=883, plot_height=615, toolbar_location='above')
      tsne = TSNE(n_components=2)
      X2d = tsne.fit_transform(np.array(D, dtype='float'))
      dsource = ColumnDataSource(
         data=dict(xcoord=X2d[:, 0], ycoord=X2d[:, 1], labels=labels))
      vocabs = LabelSet(x='xcoord', y='ycoord', text='labels', source=dsource, level='glyph',
                        render_mode='canvas')
      p.scatter(X2d[:, 0], X2d[:, 1], size=0)
      p.add_layout(vocabs)
      script, div = components(p)
      cherrypy.log("[INFO] Build word embedding plot ... OK")
      return (script, div)

   @cherrypy.expose
   def index(self):
      '''
      Default View: show a list of sentences from compaign with their type labels

      NAMES = ['Statement', 'Emotion', 'System', 'Greet',
             'Accept', 'Reject', 'whQuestion', 'Continuer',
             'ynQuestion', 'yAnswer', 'Bye', 'Clarify',
             'Emphasis', 'nAnswer', 'Other']
      :return:
      '''

      template = env.get_template('index.html')
      mail_list = self.populateMails()
      bokeh_script, bokeh_div = self.buildWordEmbedding(True)
      return template.render(mail_list=mail_list, bokeh_script=bokeh_script, bokeh_div=bokeh_div)

   @cherrypy.expose
   def plotWords(self):
      # TODO: plot word embedding view
      pass

   @cherrypy.expose
   def checktype(self, user_sents, upload_input):
      # print upload_input
      if upload_input.filename == '':
         template = env.get_template('checktypes.html')
         # load classifier first
         model_path = ''.join([self.TRAIN_MODEL_PATH, 'DlgActTypeClf.p'])
         if os.path.isfile(model_path):
            actTypeModel = dill.load(open(model_path))
         else:
            actTypeModel = self.retrainModel()
         clf = actTypeModel['clf']
         tfidfVec = actTypeModel['vectorizer']
         label_names = actTypeModel['label_names']
         use_labels = actTypeModel['user_labels']

         sents = nltk.tokenize.sent_tokenize(user_sents)
         sents_vec = tfidfVec.transform(sents)
         sents_labels = clf.predict(sents_vec)
         lin_var = clf.coef_.dot(sents_vec.T.todense()) + \
                   np.repeat(clf.intercept_.reshape(
                      clf.intercept_.size, 1), sents_labels.size, axis=1)
         sent_list = []
         for j in range(0, lin_var.shape[1]):
            sent = dict()
            sent['content'] = sents[j]
            sent['probs'] = np.array(self.softmax(lin_var[:, j])).ravel()
            prob_desc_sorted = sent['probs'].argsort()[::-1]
            sent['probs'] = ['{:.2f}'.format(
               sent['probs'][j]) for j in prob_desc_sorted]
            sent['labels'] = [label_names[use_labels[j] - 1]
                              for j in prob_desc_sorted]
            sent['idlist'] = range(len(sent['labels']))
            sent_list.append(sent)
         return template.render(message=user_sents,
                                sent_list=sent_list)
      else:
         # handle the uploaded file
         template = env.get_template('index.html')
         # upload file first
         file_uploaded = self.UPLOADED_DIR + upload_input.filename
         with open(file_uploaded, 'w+') as f:
            f.writelines(upload_input.file.readlines())
            f.flush()
         cherrypy.log("[INFO] New campagin file: {} successfully uploaded!".format(upload_input.filename))
         f.close()
         self.CAMPAIGN_FILE = file_uploaded
         self.CAMPAIGN_VECTORS = ''.join(
            [self.DATAPATH, self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.vectors.json'])
         self.CAMPAIGN_KNN = ''.join([self.DATAPATH, self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.knn.json'])
         # TODO: ajax reload
         mail_list = self.populateMails()
         bokeh_script, bokeh_div = self.buildWordEmbedding(True)
         return template.render(mail_list=mail_list, bokeh_script=bokeh_script, bokeh_div=bokeh_div)

   @cherrypy.expose
   def getSimilarWords(self, word, top_n=10):
      cherrypy.response.headers['Content-Type'] = 'application/json'
      word = word.lower()
      # return similar words based on current embedding, only support one word
      if os.path.exists(self.CAMPAIGN_KNN) is False:
         self.dumpWordDists()
      dists = json.load(open(self.CAMPAIGN_KNN, 'rb'))
      results = dict()
      rank = 0
      if dists.has_key(word):
         for w in dists[word][:top_n]:
            results[rank] = w
            rank = rank + 1
         return json.dumps(results)
      else:
         return json.dumps(results)
