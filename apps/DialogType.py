import os, re, nltk, pickle, dill, json, cherrypy
import numpy as np, jinja2 as jj2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import TreebankWordTokenizer


class Mails(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates/DialogType'))
      # NEED TO RESET PATH IF WEB RUNS ALONE
      self.TRAIN_MODEL_PATH = cherrypy.config['checkpoints_path']
      self.DATAPATH = cherrypy.config['data_path']
      # Where we save user uploaded JSON file
      self.UPLOADED_DIR = cherrypy.config['uploads_path']
      # MUST COMPLY A CERTAIN JSON FORMAT, DEFAULT WITH A SIMPLE CAMPAGIN
      # defautl JSON compaign FILE
      self.CAMPAIGN_FILE = ''.join([self.UPLOADED_DIR, 'feed.json'])
      # Pretrained Glove word embedding JSON dataset used as Lookup table
      # See more: http://nlp.stanford.edu/projects/glove/
      # self.GLOVE_SET = ''.join([self.DATAPATH, 'glove.twitter.27B.25d.json'])
      self.CAMPAIGN_VECTORS = ''.join([self.DATAPATH, self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.vectors.json'])
      # self.CAMPAIGN_KNN = ''.join([self.DATAPATH, self.CAMPAIGN_FILE.split('/')[-1].split('.')[0], '.knn.json'])

   def _softmax(self, x):
      """Compute softmax values for each sets of scores in x."""
      sf = np.exp(x)
      sf = sf / np.sum(sf, axis=0)
      return sf

   def _retrainModel(self):

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
         actTypeModel = self._retrainModel()
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
                  self._softmax(lin_var[:, j])).ravel()
               prob_desc_sorted = sent['probs'].argsort()[::-1]
               sent['probs'] = ['{:.2f}'.format(
                  sent['probs'][j]) for j in prob_desc_sorted]
               sent['labels'] = [label_names[use_labels[j] - 1]
                                 for j in prob_desc_sorted]
               sent['idlist'] = range(len(sent['labels']))
               sent_list.append(sent)
            mymail['sents'] = sent_list
            mail_list.append(mymail)

      return mail_list


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

      template = self.env.get_template('index.html')
      mail_list = self.populateMails()
      return template.render(mail_list=mail_list)

   @cherrypy.expose
   def plotWords(self):
      # TODO: plot word embedding view
      pass

   @cherrypy.expose
   def checktype(self, user_sents, upload_input):
      # print upload_input
      if upload_input.filename == '':
         template = self.env.get_template('checktypes.html')
         # load classifier first
         model_path = ''.join([self.TRAIN_MODEL_PATH, 'DlgActTypeClf.p'])
         if os.path.isfile(model_path):
            actTypeModel = dill.load(open(model_path))
         else:
            actTypeModel = self._retrainModel()

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
            sent['probs'] = np.array(self._softmax(lin_var[:, j])).ravel()
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
         template = self.env.get_template('index.html')
         # upload file first
         file_uploaded = self.UPLOADED_DIR + upload_input.filename
         with open(file_uploaded, 'w+') as f:
            f.writelines(upload_input.file.readlines())
            f.flush()
         cherrypy.log("[INFO] New campagin file: {} successfully uploaded!".format(upload_input.filename))
         f.close()
         self.CAMPAIGN_FILE = file_uploaded
         # TODO: ajax reload
         mail_list = self.populateMails()
         return template.render(mail_list=mail_list)

