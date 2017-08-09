import cherrypy, simplejson
import jinja2 as jj2
from helpers.dataset_helper import read_csv


class EvilFace(object):
   
    cherrypy.config.update({'server.socket_port': 3030,})
   
    def __init__(self):
        self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates/EvilFace/'))
        self.SAVE_DIR = cherrypy.config['uploads_path']
        self.IMG_PATH = ""
        
    @cherrypy.expose
    def index(self):
        return self.env.get_template('font3.html').render()

    
    @cherrypy.expose
    def get_info(self,img):
        # handle the saved image 
        file_uploaded = self.SAVED_DIR + img
        self.IMG_PATH = file_uploaded
        
        
        #start predicting
        
        
        
        #return informations