import cherrypy
import jinja2 as jj2

class InventoryManger(object):
   def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      self.UPLOADED_DIR = cherrypy.config['uploads_path']
      self.CSV_FILE = ''

   @cherrypy.expose
   def datasets(self):
      # handle the uploaded file
      template = self.env.get_template('InventoryManager/datalist.html')
      # TODO: read database
      example_datasets = [{"name": "dataset1",
                          "description": "description to dataset 1",
                          "downloads": 1200,
                          "owner": "huang",
                           "category": "classification",
                          "pub_date": "2016-12-10"},
                         {"name": "dataset2",
                          "description": "description to dataset 2",
                          "downloads": 2000,
                          "owner": "xiao",
                          "category": "classification",
                          "pub_date": "2016-11-10"}]
      return template.render({'apps': cherrypy.config['app_list'],
                              'datasets': example_datasets})





