'''
Data Model for registered Users

Author: Huang Xiao
Group: Cognitive Security Technologies 
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

from peewee import *

db = SqliteDatabase('database/cognuts_web_meta.db')

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    '''
    Data Model for registered User
    '''

    uid = IntegerField(null=False)
    username = CharField(null=False, unique=True)
    password = CharField(null=False)
    email = CharField(null=False)
    join_date = DateTimeField(null=False)

    class Meta:
        order_by = ('join_date',)
