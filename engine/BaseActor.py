'''
Base class for Actor based learning model.

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''


class BaseActor(object):

   def initialize(self): 
       raise NotImplementedError("Method not implemented.")

   def step(self, data_batch): 
       raise NotImplementedError("Method not implemented.")

#    def predict(self, data_batch): 
#        raise NotImplementedError("Method not implemented.")

#    def score(self, data_batch): 
#        raise NotImplementedError("Method not implemented.")

#    def save_checkpoint(self, save_ckp_path=None): 
#        raise NotImplementedError("Method not implemented.")

#    def save_parameters(self, save_param_path=None): 
#        raise NotImplementedError("Method not implemented.")

#    def listen(self, events_q): 
#        raise NotImplementedError("Method not implemented.")

#    def notify(self, events_q, payload): 
#        raise NotImplementedError("Method not implemented.")

#    def destroy(self): 
#        raise NotImplementedError("Method not implemented.")

