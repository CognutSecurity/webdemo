import cherrypy, simplejson
import jinja2 as jj2
#from helpers.dataset_helper import read_csv
#from models.H3BayesNet import H3BayesNet

class StackSecure(object):
    def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      
    @cherrypy.expose
    def index(self):
        app_list = cherrypy.config['app_list']
        return self.env.get_template('StackSecure/index.html').render({"apps": app_list, "page_subtitle": "Secure Stack Code Snippets"})
