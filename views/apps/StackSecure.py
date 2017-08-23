import cherrypy, simplejson
import jinja2 as jj2
import sqlite3
from helpers.dataset_helper import read_csv

# TODO delete snippets and use the database and the users related table
# snippets = []

class StackSecure(object):
    def __init__(self):
      self.env = jj2.Environment(loader=jj2.FileSystemLoader('./templates'))
      # TODO delete snippets and use the database and the users related table
      self.snippets = []
      
    @cherrypy.expose
    def index(self):
        app_list = cherrypy.config['app_list']
        print cherrypy.request.config

        return self.env.get_template('StackSecure/index.html').render({"apps": app_list, "page_subtitle": "Secure Stack Code Snippets"})

@cherrypy.expose
class SecurityAnalysisWebService(object):
    
    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        # return current data in here
        return cherrypy.session['security']

    def POST(self, snippet):
        # modification of data dependent on POST input parameters in here
        cherrypy.session['security'] = '1'
        return cherrypy.session['security']

    def PUT(self, security):
        # init code and data replacement in here
        cherrypy.session['security'] = security

    def DELETE(self):
        # delete data in here
        cherrypy.session.pop('security', None)

@cherrypy.expose
class SnippetWebService(object):
    def __init__(self, stacksecure):
        self.stack_secure = stacksecure

    def GET(self):
        return simplejson.dumps({'snippet' : cherrypy.session['snippet']}) 
    
    def POST(self):
        cherrypy.session['snippet'] = cherrypy.session['snippets'].pop()

        return simplejson.dumps({"snippet" : cherrypy.session['snippet']}) 

    def PUT(self):
        # TODO replace code using the database and the users related table
        cherrypy.session['snippets'] = self.stack_secure.snippets
        cherrypy.session['snippet'] = cherrypy.session['snippets'].pop()

        return simplejson.dumps({}) 

        #with sqlite3.connect(DB_STRING) as c:
        #    snippets = []
        #    r = c.execute('SELECT snippet FROM snippets')
        #    rows = r.fetchall()
        #    for row in rows:
        #        snippets.append(row)

        #    cherrypy.session['snippets'] = snippets
        #    cherrypy.session['snippet'] = snippets.pop()

    def DELETE(self):
        cherrypy.session.pop('snippets', None)

@cherrypy.expose
class SimilarSnippetWebService(object):

    def GET(self):
        return simplejson.dumps({'snippet' : cherrypy.session['similarsnippet']}) 

    def POST(self):
        cherrypy.session['similarsnippet'] = "// other similar snippet"
        
        return simplejson.dumps({'snippet' : cherrypy.session['similarsnippet']}) 

    def PUT(self, snippet):
        cherrypy.session['similarsnippet'] = "// similar snippet"

        return simplejson.dumps({})

    def DELETE(self):
        cherrypy.session.pop('similarsnippet', None)

@cherrypy.expose
class SnippetDatabaseHelperWebService(object):

    def __init__(self, stacksecure):
      self.UPLOADED_DIR = cherrypy.config['uploads_path']
      self.stack_secure = stacksecure
    
    def PUT(self, upload_input):
        # TODO replace with database code
        self.stack_secure.snippets = []
        import csv
        file_uploaded = self.UPLOADED_DIR + upload_input.filename
        with open(file_uploaded, 'rb') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                self.stack_secure.snippets.append(row[2])
        f.close()

        #with sqlite3.connect(DB_STRING) as con:
        #    con.execute("DROP TABLE IF EXISTS snippets")
        #    con.execute("CREATE TABLE snippets (snippet TEXT)")
        #    con.execute("COPY snippets FROM file_uploaded CSV HEADER")

    def DELETE(self):
        # TODO replace code using the database and the users related table
        self.stack_secure.snippets = []
        
        #with sqlite3.connect(DB_STRING) as con:
        #    con.execute("DROP TABLE IF EXISTS snippets")

def setup_database():
    print "setup db"

def cleanup_database():
    print "cleanup db"

cherrypy.engine.subscribe('start', setup_database)
cherrypy.engine.subscribe('stop', cleanup_database)
