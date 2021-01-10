# -*- coding: utf-8 -*-
"""
Project Repo for Nudge 

@author: Bennimi
"""

# init all .py packages

__all__ = ['TextProcessor']


import datetime as datetime
def timestemp():
    return str(datetime.datetime.now().strftime("%Y%m%d_%H:%M"))