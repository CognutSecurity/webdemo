'''
Data Model for registered Users

Author: Huang Xiao
Group: Cognitive Security Technologies 
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

from peewee import CharField, DateTimeField, Model

class User(Model):
    '''
    Data Model for registered User
    '''

    username = CharField(null=False, unique=True)
    password = CharField(null=False)
    email = CharField(null=False)
    join_date = DateTimeField(null=False)

    class Meta:
        order_by = ('join_date',)
