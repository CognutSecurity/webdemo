import cherrypy, simplejson
import jinja2 as jj2
from orm.Datasets import Dataset
from peewee import SqliteDatabase, DoesNotExist, DatabaseError
from datetime import datetime


class InventoryManager(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      self.UPLOADED_DIR = cherrypy.config['data_path']
      # self.CSV_FILE = ''
      self._db = SqliteDatabase(cherrypy.config['sql_db'])
      self._db.connect()

   @cherrypy.expose
   def datasets(self):
      # check permission
      if "logged_user" in cherrypy.session:
         owner = cherrypy.session['logged_user']
      else:
         owner = 'public'
      template = self.env.get_template('InventoryManager/datalist.html')
      # read database
      Dataset._meta.database = self._db
      available_datasets = []
      for dataset in Dataset.select().where(Dataset.owner == owner):
         available_datasets.append({"name": dataset.name,
                                    "description": dataset.description,
                                    "downloads": dataset.downloads,
                                    "owner": dataset.owner_id,
                                    "pub_date": dataset.pub_date})
      return template.render({'apps': cherrypy.config['app_list'],
                              'available_datasets': available_datasets})

   @cherrypy.expose
   def checkpoints(self):
      pass

   @cherrypy.expose
   def upload(self, upload_input):
      # check permission
      if "logged_user" in cherrypy.session:
         owner = cherrypy.session['logged_user']
      else:
         owner = 'public'

      # connect db
      Dataset._meta.database = self._db
      error_msg = dict()
      try:
         Dataset.get(Dataset.name == upload_input.filename)
         error_msg['stats'] = 101
         error_msg['info'] = 'Dataset {:s} already exists!'.format(upload_input.filename)
         self._db.close()
         return simplejson.dumps(error_msg)
      except DoesNotExist:
         # handle the uploaded file
         file_uploaded = self.UPLOADED_DIR + upload_input.filename
         with open(file_uploaded, 'w+') as f:
            f.writelines(upload_input.file.readlines())
            f.flush()
         cherrypy.log("[INFO] New dataset: {} successfully uploaded!".format(upload_input.filename))
         f.close()
         # insert to database
         try:
            Dataset.create(name=upload_input.filename,
                           path=file_uploaded,
                           owner=owner,
                           pub_date=datetime.now())
            self._db.close()
            error_msg['stats'] = 100
            error_msg['info'] = 'Data successfully uploaded.'
            return simplejson.dumps(error_msg)
         except DatabaseError as err:
            error_msg['stats'] = 101
            error_msg['info'] = 'Error occurs while saving data, try again!'
            cherrypy.log(err.message)
            return simplejson.dumps(error_msg)
