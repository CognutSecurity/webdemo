# -*- coding: utf-8 -*-

import os
import time
import threading
import cherrypy

from helpers import config, camera


import os
import uuid
import zipfile
import StringIO




class EvilFace(object):
    def __init__(self, camera, html, folder, count):
        self.working = True
        self.camera = camera
        self.html = html.strip()

        self.folder = folder
        self.count = count

        cherrypy.engine.subscribe("stop", self.stop)

    def stop(self):
        self.working = False

    # noinspection PyUnusedLocal
    @cherrypy.expose
    def default(self, *args, **kwargs):
        return self.html

    # noinspection PyUnusedLocal
    @cherrypy.expose
    def image(self, *args, **kwargs):
        cherrypy.response.headers['Content-Type'] = "text/html"
        cherrypy.response.headers['Pragma'] = "no-cache"

        return "<img src=\"data:image/jpeg;base64,%s\">" % self.camera.getBase64Image()

    # noinspection PyUnusedLocal
    @cherrypy.expose
    def images(self, *args, **kwargs):
        cherrypy.response.headers['Content-Type'] = "application/zip"
        cherrypy.response.headers['Pragma'] = "no-cache"

        cherrypy.response.headers['Content-Disposition'] = "attachment; filename=\"%s.zip\"" % str(uuid.uuid1())[:8]

        zipOutput = StringIO.StringIO()
        zipContainer = zipfile.ZipFile(zipOutput, "w")

        for root, _, images in os.walk(self.folder):
            for i, image in enumerate(sorted(images, key=lambda img: os.stat(os.path.join(root, img)).st_mtime, reverse=True)):
                if i < self.count:
                    zipContainer.write(os.path.join(root, image), image)

        zipContainer.close()

        return zipOutput.getvalue()



if not os.path.exists(config.camera['folder']):
    os.makedirs(config.camera['folder'])
    


###cameraInstance = camera.Camera(config.camera['index'], config.camera['width'], config.camera['height'])
###if config.camera['interval'] > 0:
   ### removeOldImages(config.cleaner['interval'], config.camera['folder'], config.cleaner['old'])
   ### storeImageByTimer(cameraInstance, config.camera['interval'], config.camera['folder'])

    #cherrypy.config.update({
    #    "engine.autoreload.on": False,
    #    "server.socket_host": config.server['host'],
    #    "server.socket_port": config.server['port']
    #})
    cherrypy.config.update({'server.socket_port': 3030,})
###cherrypy.quickstart(EvilFace(cameraInstance, config.server['html'], config.camera['folder'], config.camera['count']), "/", config={"/": {}})
