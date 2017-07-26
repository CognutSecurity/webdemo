import cherrypy
import jinja2 as jj2

class Home(object):

   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))

   @cherrypy.expose
   def index(self):
      app_list = cherrypy.config['app_list']
      return self.env.get_template('Home/index.html').render({"apps": app_list,
                                                          "page_subtitle": "Home"})

   @cherrypy.expose
   def signup(self):
      return self.env.get_template('signup.html').render(page_subtitle="Sign Up")

   @cherrypy.expose
   def login(self):
      return self.env.get_template('login.html').render(page_subtitle="Login")

   @cherrypy.expose
   def reset(self):
      return self.env.get_template('reset.html').render(page_subtitle="Reset Password")