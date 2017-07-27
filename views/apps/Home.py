import cherrypy
import jinja2 as jj2
from peewee import *
from DataModels.Users import User
from datetime import datetime


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

   @cherrypy.expose
   def register(self, uname, email, password, retype_password, agree):
      db = SqliteDatabase(cherrypy.config['sql_db'])
      User._meta.database = db
      db.connect()
      new_user = User(username = uname,
                      email = email,
                      password = password,
                      join_date = datetime.now())
      new_user.save(force_insert=True)
      db.close()
      cherrypy.HTTPRedirect('/')

