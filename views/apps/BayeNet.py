import cherrypy, simplejson
import jinja2 as jj2
from helpers.dataset_helper import read_csv
from models.H3BayesNet import H3BayesNet
from peewee import DoesNotExist, SqliteDatabase
from orm.Datasets import Dataset

class BayesNet(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      self.DATA_DIR = cherrypy.config['data_path']
      self.CSV_FILE = ''
      self._db = SqliteDatabase(cherrypy.config['sql_db'])
      self._db.connect()

   @cherrypy.expose
   def index(self):
      resp = dict()
      # check permission
      if "logged_user" in cherrypy.session:
         owner = cherrypy.session['logged_user']
         resp['logged_user'] = owner
      else:
         owner = 'public'
      # read database
      Dataset._meta.database = self._db
      available_datasets = []
      for dataset in Dataset.select().where(Dataset.owner == owner):
         available_datasets.append({"name": dataset.name,
                                    "description": dataset.description,
                                    "downloads": dataset.downloads,
                                    "owner": dataset.owner_id,
                                    "pub_date": dataset.pub_date})
      # render view
      resp['apps'] = cherrypy.config['app_list']
      resp['page_subtitle'] = "BayesNetwork"
      resp['available_datasets'] = available_datasets
      return self.env.get_template('BayesNet/index.html').render(resp)

   @cherrypy.expose
   def draw(self, datafile_name, alpha, method, penalty, bin, pval):
      # get the data from store
      Dataset._meta.database = self._db
      error_msg = dict()
      try:
         data = Dataset.get(Dataset.name == datafile_name)
         # handle the uploaded file
         Xtr, nodes = read_csv(data.path)
         clf = H3BayesNet(alpha=float(alpha), vnames=nodes, method=method, penalty=float(penalty), bin=int(bin), pval=float(pval))
         clf.fit(Xtr)
         ci_coef = clf.ci_coef_
         adjmat = clf.conditional_independences_
         edges = self._get_e(ci_coef, adjmat, nodes)
         return simplejson.dumps({"edges": edges,
                                  "node_names": nodes})
      except DoesNotExist:
         error_msg['stats'] = 101
         error_msg['info'] = 'Requested dataset does not exit!'
         return simplejson.dumps(error_msg)

   def _get_e(self, coef, adjmat, node_names):
      # parse coef and adjacent matrix into a cytoscape-compatible dist
      node_size = coef.shape[0]
      e = []
      for i in range(node_size):
         for j in range(i+1, node_size):
            if adjmat[i, j] == 1 and adjmat[j, i] == 0:
               e.append({
                  "id": coef[i, j], "source": node_names[i], "target": node_names[j]
               })
            if adjmat[i, j] == 0 and adjmat[j, i] == 1:
               e.append({
                  "id": coef[j, i], "source": node_names[j], "target": node_names[i]
               })
            if adjmat[i, j] == 1 and adjmat[j, i] == 1:
               e.append({
                  "id": coef[i, j], "source": node_names[i], "target": node_names[j], "directed": False
               })
      return e





