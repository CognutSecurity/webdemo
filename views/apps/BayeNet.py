import cherrypy
import jinja2 as jj2
from mlcore.utils.dataset_helper import read_csv
from mlcore.models.H3BayesNet import H3BayesNet


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
      template = self.env.get_template('index.html')
      # upload file first
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
      # TODO: ajax reload
      return template.render(ci_coef=ci_coef, adjmat=adjmat)
