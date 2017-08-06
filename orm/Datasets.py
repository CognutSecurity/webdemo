'''
Data Model for existing datasets
which can either be user-uploaded or system built-in toy samples.

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

from peewee import CharField, ForeignKeyField, DateTimeField, IntegerField, Model
from Users import User


class Dataset(Model):
   '''
   Data Model for registered User
   '''

   name = CharField(null=False, unique=True)
   path = CharField(null=False)
   owner = ForeignKeyField(User, to_field="username", on_delete="CASCADE", on_update="CASCADE", null=False)
   description = CharField(default='No description')
   downloads = IntegerField(default=0)
   pub_date = DateTimeField(null=False)

   class Meta:
      order_by = ('owner',)
