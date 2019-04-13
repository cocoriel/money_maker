# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import pymongo
import numpy as np
import statsmodels.tsa.stattools as ts
from datetime import datetime, timedelta, date
import getData
import stuff

### p.115
def get_hurst_exponent(df):
    lags = range(2, 100)
    ts = np.log(df)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    result = poly[0]*2.0
    return result

### p.117
def get_half_life(df):
    price = pd.Series(df)
    lagged_price = price.shift(1).fillna(method = "bfill")
    delta = price - lagged_price
    beta = np.polyfit(lagged_price, delta, 1)[0]
    half_life = (-1*np.log(2)/beta)
    return half_life


def main():
    stock_code = getData.mongo().get_stock_code()
    print("Stock Code :: len=%d" %(len(stock_code)))

    ## 분석데이터 넣기
    anly = stock_code[['_id', 'company', 'suffix']].copy(deep=True)
    anly['code'] = anly['_id'] + '.' + anly['suffix']
    anly['adf_result']          = -1
    anly['is_avgmv_by_adf']     = -1
    anly['is_avgmv_by_hurst']   = -1
    anly['is_trend_by_hurst']   = -1
    anly['half_life']           = -1
    #### for
    code = "000720.KS" #stock_code[0]  ## for test
    print("code:%s" %(code))

    price = getData.mongo().get_stock_price_by_code(code)
    price = stuff.utils().cal_moving_average(price)

    ### 분석 결과 (code당 1개씩 나오는것)
    adf_result = ts.adfuller(price['Close'])
    hurst_result = get_hurst_exponent(price['Close'])

    ## 평균회귀를 하는지 판단.
    is_avgmv_by_adf = True if sum(adf_result[0] < adf_result[4].values()) > 0 else False
    is_avgmv_by_hust = hurst_result <= 0.1  ## 0에 가까울수록, 평균회귀성향
    is_trend_by_hust = hurst_result >= 0.9  ## 1에 가까울수록, 추세성향

    ## 평균회귀의 half-lite - 작다는것은 주식값의 변동이 잦다(자주발생)
    half_life = get_half_life(price['Close'])

    ## 결과값 넣기
    index = anly[anly.code == code].index[0]
    anly.at[index, 'adf_result']          = adf_result
    anly.at[index, 'is_avgmv_by_adf']     = is_avgmv_by_adf
    anly.at[index, 'is_avgmv_by_hurst']   = is_avgmv_by_hust
    anly.at[index, 'is_trend_by_hurst']   = is_trend_by_hust

## (참고) python 3항연산자 : True if condition else False
