# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import pymongo
import numpy as np
import statsmodels.tsa.stattools as ts
from datetime import datetime, timedelta, date


class utils:

    def __init__(self):
        pass

    def _normalize(self, data):
        normalized_data = (data-data.min())/(data.max()-data.min())
        return normalized_data

    def cal_moving_average(self, price):
        for period in [5, 10, 20, 60, 120]:
            ## rolling_mean(5) : 나를 포함한 5개의 평
            price[('CloseM%dMean' %(period))] = price['Close'].rolling(window=period, center=False).mean()
            price[('CloseM%dStd'  %(period))] = price['Close'].rolling(window=period, center=False).std()
            price[('CloseM%dMeanNorm' %(period))] = self._normalize(price[('CloseM%dMean' %(period))] )
            price[('CloseM%dStdNorm' %(period))] = self._normalize(price[('CloseM%dStd' %(period))] )

        for period in [10, 20, 60, 120]:
            price[('VolumeM%dMean' %(period))] = price['Volume'].rolling(window=period, center=False).mean()
            price[('VolumeM%dStd'  %(period))] = price['Volume'].rolling(window=period, center=False).std()
            price[('VolumeM%dMeanNorm' %(period))] = self._normalize(price[('VolumeM%dMean' %(period))] )
            price[('VolumeM%dStdNorm' %(period))] = self._normalize(price[('VolumeM%dStd' %(period))] )

        return price