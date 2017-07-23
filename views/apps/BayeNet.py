import cherrypy, simplejson
import jinja2 as jj2
from helpers.dataset_helper import read_csv
from models.H3BayesNet import H3BayesNet

class BayesNet(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates/BayesNet/'))
      self.UPLOADED_DIR = cherrypy.config['uploads_path']
      self.CSV_FILE = ''

   @cherrypy.expose
   def index(self):
      return self.env.get_template('index.html').render()

   @cherrypy.expose
   def draw(self, upload_input):
      # handle the uploaded file
      file_uploaded = self.UPLOADED_DIR + upload_input.filename
      with open(file_uploaded, 'w+') as f:
         f.writelines(upload_input.file.readlines())
         f.flush()
      cherrypy.log("[INFO] New dataset .csv file: {} successfully uploaded!".format(upload_input.filename))
      f.close()
      self.CSV_FILE = file_uploaded
      Xtr, nodes = read_csv(self.CSV_FILE)
      clf = H3BayesNet(alpha=0.1, vnames=nodes, method='ledoit_wolf', penalty=1.0)
      clf.fit(Xtr)
      ci_coef = clf.ci_coef_
      adjmat = clf.conditional_independences_
      edges = self._get_e(ci_coef, adjmat, nodes)
      return simplejson.dumps({"edges": edges,
                               "node_names": nodes})

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





