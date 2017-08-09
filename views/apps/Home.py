import cherrypy, simplejson
import jinja2 as jj2
from peewee import *
from orm.Users import User
from datetime import datetime


class Authentication(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      self._db = SqliteDatabase(cherrypy.config['sql_db'])
      self._db.connect()

   @cherrypy.expose
   def signup(self):
      # TODO: ajax handle error msg
      return self.env.get_template('signup.html').render(page_subtitle="Sign Up")

   @cherrypy.expose
   def login(self):
      # TODO: ajax handle error msg
      return self.env.get_template('login.html').render(page_subtitle="Login")

   @cherrypy.expose
   def reset(self):
      # TODO: email reset password
      return self.env.get_template('reset.html').render(page_subtitle="Reset Password")

   @cherrypy.expose
   def register(self, uname, email, password, retype_password, agree):
      # TODO: ajax handle error msg
      User._meta.database = self._db
      error_msg = dict()
      try:
         User.get(User.username == uname)
         error_msg['username'] = 'User {:s} already exists!'.format(uname)
         return simplejson.dumps(error_msg)
      except DoesNotExist:
         try:
            User.get(User.email == email)
            error_msg['email'] = 'Email {:s} has been already taken!'.format(email)
            return simplejson.dumps(error_msg)
         except DoesNotExist:
            # registered data complies DB
            User.insert(username=uname,
                        email=email,
                        password=password,
                        join_date=datetime.now()).execute()
            cherrypy.session['logged_user'] = uname
            self._db.close()
            raise cherrypy.HTTPRedirect('/')

   @cherrypy.expose
   def do_auth(self, username, password):
      # TODO: ajax handle error msg
      User._meta.database = self._db
      error_msg = dict()
      try:
         u = User.get(User.username == username)
         if u.password == password:
            cherrypy.session['logged_user'] = username
            # TODO: ajax redirect
            raise cherrypy.HTTPRedirect('/')
         else:
            error_msg['info'] = 'Either user or password is not correct!'
            return simplejson.dumps(error_msg)
      except DoesNotExist:
         error_msg['username'] = 'User {:s} does not exist!'.format(username)
         return simplejson.dumps(error_msg)



class Home(object):

   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      self.auth = Authentication()

   @cherrypy.expose
   def index(self):
      app_list = cherrypy.config['app_list']
      if "logged_user" in cherrypy.session:
         return self.env.get_template('Home/index.html').render({"apps": app_list,
                                                                 "page_subtitle": "Home",
                                                                 "logged_user": cherrypy.session['logged_user']})
      else:
         return self.env.get_template('Home/index.html').render({"apps": app_list,
                                                                 "page_subtitle": "Home",
                                                                 "logged_user": ''})

   @cherrypy.expose
   def ws(self):
      '''You can not send yet, because handshake has not been completed'''
      cherrypy.log('Handler created: %r' % cherrypy.request.ws_handler)
