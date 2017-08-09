from ws4py.websocket import WebSocket
from ws4py.messaging import TextMessage
import cherrypy, json

class WebSocketHandler(WebSocket):
   socket_set = set()

   def received_message(self, message):
      msg = json.loads(message.data.decode())
      self.send('echo:' + msg['cmd'])

   def opened(self):
      cherrypy.log('web socket opened!')
      self.socket_set.add(self)

   def closed(self, code, reason=None):
      cherrypy.log('web socket closed!')
      self.socket_set.remove(self)