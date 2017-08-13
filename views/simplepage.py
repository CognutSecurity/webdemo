import cherrypy
from apps.DialogType import Mails
from apps.BayeNet import BayesNet
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
cameraInstance = camera.Camera(config.camera['index'], config.camera['width'], config.camera['height'])
if config.camera['interval'] > 0:
        camera.removeOldImages(config.cleaner['interval'], config.camera['folder'], config.cleaner['old'])
        camera.storeImageByTimer(cameraInstance, config.camera['interval'], config.camera['folder'])

cherrypy.tree.mount(EvilFace(cameraInstance, config.server['html'], config.camera['folder'], config.camera['count']), "/evilface", config={"/": {}})

# initialize necessary databases
db = init_db()

# setup WebSocket support
WebSocketPlugin(cherrypy.engine).subscribe()
cherrypy.tools.websocket = WebSocketTool()

# start cherrypy bus and server
cherrypy.engine.start()
cherrypy.engine.block()
