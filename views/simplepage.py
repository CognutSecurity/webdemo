import cherrypy
from apps.DialogType import Mails
from apps.BayesNet import BayesNet
from apps.StackSecure import StackSecure
from apps.InventoryManager import InventoryManger
from apps.Home import Home
from apps.WebSocketHandler import WebSocketHandler
from peewee import *
from orm.Users import User
from orm.Datasets import Dataset
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool


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

# mount apps
cherrypy.config.update(config='confs/global.cfg')
cherrypy.tree.mount(Home(), '/', config='confs/default.cfg')
cherrypy.tree.mount(Mails(), '/mails', config='confs/DialogType.cfg')
cherrypy.tree.mount(BayesNet(), '/bayesnet', config='confs/BayesNet.cfg')
cherrypy.tree.mount(StackSecure(), '/stacksecure', config='confs/StackSecure.cfg')
cherrypy.tree.mount(InventoryManger(), '/inventory', config='confs/InventoryManager.cfg')

# TODO: better logging
# enhancement goes here...

# initialize necessary databases for the first time
db = init_db()

# setup WebSocket support for cherrypy
WebSocketPlugin(cherrypy.engine).subscribe()
cherrypy.tools.websocket = WebSocketTool()

# start cherrypy bus and server
cherrypy.engine.start()
cherrypy.engine.block()
