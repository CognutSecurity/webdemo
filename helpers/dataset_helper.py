"""
This utility file contains helper functions for dataset preparation. It is intended for preparing
datasets used in other places.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import json, numpy as np

def get_data_batch(events_json):
   '''
   Return a dataset in a single batch from events json file
   :param events_json: filename of events JSON
   :return: dict contain dataset
   '''

   fd = open(events_json, 'r')
   content = fd.read()
   data = json.loads(content)
   res = dict()
   cnt_in = 0
   cnt_out = 0
   template_list = set()
   for row in data:
      ACTION = row[0]
      TIMESTAMP = row[1]
      CAMPAIGN_ID = row[2]
      EMAIL_IDs = row[3]
      CONTENT = row[4]

      # form the dictionary
      if ACTION == 'IN':
         # check if the keys exist
         if not res.has_key(CAMPAIGN_ID):
            res[CAMPAIGN_ID] = dict()
         if not res[CAMPAIGN_ID].has_key(EMAIL_IDs):
            res[CAMPAIGN_ID][EMAIL_IDs] = dict()
         # update values
         res[CAMPAIGN_ID][EMAIL_IDs]['timestamp'] = TIMESTAMP
         res[CAMPAIGN_ID][EMAIL_IDs]['content'] = CONTENT
         cnt_in += 1
      if ACTION == 'TEMPLATE':
         for eid in EMAIL_IDs:
            # check if the keys exist
            if not res.has_key(CAMPAIGN_ID):
               res[CAMPAIGN_ID] = dict()
            if not res[CAMPAIGN_ID].has_key(eid):
               res[CAMPAIGN_ID][eid] = dict()
            res[CAMPAIGN_ID][eid]['template'] = CONTENT
            cnt_out += 1
            if CONTENT:
               template_list.add(CONTENT)
   meta = {'received': cnt_in,
           'sent': cnt_out,
           'templates_used': list(template_list)}
   res['meta'] = meta
   print '[INFO] Total received {:d} mails and sent {:d} mails with {:d} different templates.'.format(cnt_in, cnt_out,
                                                                                                      len(
                                                                                                         template_list))
   return res


def get_data_set(events_json):
   '''
   Get a trainable dataset from the events JSON file, note that only emails with template ID are returned
   :param events_json: filename of events JSON
   :return: ndarray with last col as labels
   '''

   data = get_data_batch(events_json)
   emails = []
   labels = []
   for k in data.keys():
      if k == 'meta':
         continue
      cid = k
      email_ids = data[k].keys()
      for id in email_ids:
         if data[k][id].has_key('template') and data[k][id]['template']:
            # only save the data with valid template id
            emails.append(data[k][id]['content'])
            labels.append(data[k][id]['template'])
   dataset = dict()
   dataset['features'] = emails
   dataset['targets'] = labels
   print '[INFO] Obtained {:0} emails with template responses.'.format(len(labels))
   return dataset


def read_csv(filename, header_line=True):
   '''
   load dataset from csv file
   :param filename: file path to csv
   :return: training features, a list of node names
   '''

   import csv

   csv_file = open(filename, 'r')
   reader = csv.reader(csv_file)
   cnt = 0
   input_data = []
   nodes = []
   for row in reader:
      if cnt == 0 and header_line:
         nodes = row
      else:
         input_data.append([float(elem) for elem in row])
      cnt += 1
   csv_file.close()
   return np.array(input_data), nodes


def csv2dict(csv):
   '''
   convert csv data to a list of dicts
   :return: dict
   '''

   data, nodes = read_csv(csv)
   res = list([])
   for row_id in range(data.shape[0]):
      instance = dict()
      for i, n in enumerate(nodes):
         instance[n] = data[row_id, i]
      res.append(instance)
   return res
