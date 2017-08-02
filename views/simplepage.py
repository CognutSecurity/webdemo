import cherrypy
from apps.DialogType import Mails
from apps.BayeNet import BayesNet
from apps.InventoryManager import InventoryManger
from apps.Home import Home
from peewee import *
from DataModels.Users import User
from DataModels.Datasets import Dataset


def init_db():
   # initialiate databases if they are not exist
   db = SqliteDatabase(cherrypy.config['sql_db'])
   User._meta.database = db
   Dataset._meta.database = db
   try:
      db.connect()
      db.create_tables([User, Dataset], safe=True)
      db.close()
   except DatabaseError:
      cherrypy.log('Database error happens.')
   return db

# initialize necessary databases
cherrypy.config.update(config='confs/global.cfg')
cherrypy.tree.mount(Home(), '/', config='confs/default.cfg')
cherrypy.tree.mount(Mails(), '/mails', config='confs/DialogType.cfg')
cherrypy.tree.mount(BayesNet(), '/bayesnet', config='confs/BayesNet.cfg')
cherrypy.tree.mount(InventoryManger(), '/inventory', config='confs/InventoryManager.cfg')
db = init_db()
cherrypy.engine.start()
cherrypy.engine.block()
