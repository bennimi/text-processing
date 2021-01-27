# -*- coding: utf-8 -*-
"""

@author: Bennimi
"""

# init all .py 

__all__ = ['TextProcessor']


import datetime as datetime
def timestemp():
    return str(datetime.datetime.now().strftime("%Y%m%d_%H:%M"))