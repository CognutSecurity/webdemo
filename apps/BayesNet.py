import cherrypy
import simplejson
import jinja2 as jj2
import numpy as np
from h3mlcore.utils.DatasetHelper import read_csv
from h3mlcore.models.H3BayesNet import H3BayesNet
from peewee import DoesNotExist, SqliteDatabase
from orm.Datasets import Dataset
from ws4py.websocket import WebSocket
from sklearn.manifold import TSNE
import os.path


class BayesNet(object):
    def __init__(self):
        self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
        self.DATA_DIR = cherrypy.config['data_path']
        self.CSV_FILE = ''
        self.db = SqliteDatabase(cherrypy.config['sql_db'])
        self.db.connect()

    @cherrypy.expose
    def ws(self):
        # naive websocket zone
        cherrypy.log('[WEBSOCKETS] Handler created: %r' % cherrypy.request.ws_handler)
        pass

    @cherrypy.expose
    def index(self):
        resp = dict()
        # check permission
        if "logged_user" in cherrypy.session:
            owner = cherrypy.session['logged_user']
            resp['logged_user'] = owner
        else:
            owner = 'public'

        # read database
        Dataset._meta.database = self.db
        available_datasets = []
        for dataset in Dataset.select().where(Dataset.owner == owner):
            available_datasets.append({"name": dataset.name,
                                       "description": dataset.description,
                                       "downloads": dataset.downloads,
                                       "owner": dataset.owner_id,
                                       "pub_date": dataset.pub_date})
        # render view
        resp['apps'] = cherrypy.config['app_list']
        resp['page_subtitle'] = "BayesNetwork"
        resp['available_datasets'] = available_datasets
        return self.env.get_template('BayesNet/index.html').render(resp)

    class SocketHandler(WebSocket):
        '''
        App handler for websocket. This opens up a live socket for asyn. callbacks
        '''

        socket_set = set()

        def received_message(self, message):
            '''
            callback to handle coming message
            :param message: JSON object
            :return:
            '''

            msg = simplejson.loads(message.data.decode())

            if msg['cmd'] == 'train':
                error_msg = dict()
                Dataset._meta.database = SqliteDatabase(
                    cherrypy.request.config['sql_db'])
                Dataset._meta.database.connect()

                try:
                    data = Dataset.get(
                        Dataset.name == msg['formData']['datafile_name'])
                    # learn the causal structure by CBN
                    Xtr, nodes = read_csv(data.path)
                    clf = H3BayesNet(alpha=float(msg['formData']['alpha']),
                                     vnames=nodes,
                                     method=msg['formData']['method'],
                                     penalty=float(msg['formData']['penalty']),
                                     bin=int(msg['formData']['bin']),
                                     pval=float(msg['formData']['pval']),
                                     samplesize=int(msg['formData']['ssample']),
                                     messager=self,
                                     log_config='logging.yaml',
                                     verbose=False)
                    clf.fit(Xtr)
                    ci_coef = clf.ci_coef_
                    adjmat = clf.conditional_independences_
                    edges = self._get_e(ci_coef, adjmat, nodes)

                    # dimension reduction
                    tsne = TSNE(n_components=2, )
                    if int(msg['formData']['ssample']) >= Xtr.shape[0] or int(msg['formData']['ssample']) == 0:
                        X_proj = tsne.fit_transform(Xtr)
                    else:
                        X_proj = tsne.fit_transform(Xtr[np.random.choice(
                            Xtr.shape[0], int(msg['formData']['ssample']))])
                    sample_dist = {
                        'x': X_proj[:, 0].tolist(), 'y': X_proj[:, 1].tolist()}
                    self.send(simplejson.dumps(
                        {"edges": edges, "node_names": nodes, "samples2d": sample_dist, "stats": 100}))

                except DoesNotExist:
                    error_msg['stats'] = 101
                    error_msg['info'] = 'Requested dataset does not exit!'
                    self.send(simplejson.dumps(error_msg))

        def opened(self):
            cherrypy.log(__file__.split(
                '/')[-1] + ':' + self.__class__.__name__ + ': WebSocket %{:s} opened!'.format(str(self.sock)))
            self.socket_set.add(self)

        def closed(self, code, reason=None):
            cherrypy.log(__file__.split(
                '/')[-1] + ':' + self.__class__.__name__ + ': WebSocket %{:s} closed!'.format(str(self.sock)))
            self.socket_set.remove(self)

        def _get_e(self, coef, adjmat, node_names):
            '''
            parse coef and adjacent matrix into a cytoscape-compatible dist
            '''

            node_size = coef.shape[0]
            e = []
            for i in range(node_size):
                for j in range(i + 1, node_size):
                    if adjmat[i, j] == 1 and adjmat[j, i] == 0:
                        e.append({
                            "id": coef[i, j], "source": node_names[i], "target": node_names[j]
                        })
                    if adjmat[i, j] == 0 and adjmat[j, i] == 1:
                        e.append({
                            "id": coef[j, i], "source": node_names[j], "target": node_names[i]
                        })
                    if adjmat[i, j] == 1 and adjmat[j, i] == 1:
                        e.append({
                            "id": coef[i, j], "source": node_names[i], "target": node_names[j], "directed": False
                        })
            return e
