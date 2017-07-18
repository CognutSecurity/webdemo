import os, cherrypy
from apps.DialogType import Mails
from apps.BayeNet import BayesNet
from apps.Home import Home


if __name__ == '__main__':
   cherrypy.config.update(config='confs/global.cfg')
   cherrypy.tree.mount(Home(), '/', config='confs/default.cfg')
   cherrypy.tree.mount(Mails(), '/mails', config='confs/DialogType.cfg')
   cherrypy.tree.mount(BayesNet(), '/bayesnet', config='confs/BayesNet.cfg')
   cherrypy.engine.start()
   cherrypy.engine.block()
