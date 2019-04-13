# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import pymongo
import numpy as np
import statsmodels.tsa.stattools as ts
from datetime import datetime, timedelta, date


class mongo:
    def __init__(self):
        pass


    def _connect(self):
        client = pymongo.MongoClient(
            host        = "ec2-13-125-126-72.ap-northeast-2.compute.amazonaws.com",
            port        = 27017,
            username    = "root",
            password    = "example"
            #authSource  = "money",
        )
        return client

    def get_stock_code(self):
        client = self._connect()

        db = client['money']
        collection = db['stock-code']
        post = collection.find({})
        stock_code = pd.DataFrame(list(post))

        client.close()

        return stock_code


    def get_stock_price_by_code(self, code):
        client = self._connect()

        db = client['money']
        collection = db['stock-price']
        post = collection.find({'Code' : code})
        stock_price = pd.DataFrame(list(post))

        client.close()

        return stock_price
    '''
    def get_stock_price_by_date(self, start_date, end_date):
        client = self._connect()

        db = client['money']
        collection = db['stock-price']
        ### todo 
        #post = collection.find({'Code' : code})
        #stock_price = pd.DataFrame(list(post))

        client.close()

        return stock_price
    '''

    def get_stock_price_all(self):
        client = self._connect()

        db = client['money']
        collection = db['stock-price']
        post = collection.find({})
        stock_price = pd.DataFrame(list(post))

        client.close()

        return stock_price