import cherrypy
import jinja2 as jj2

class Home(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates/Home'))

   @cherrypy.expose
   def index(self):
      app_list = cherrypy.request.app.config['APPS']['app_list']
      return self.env.get_template('index.html').render(apps=app_list)