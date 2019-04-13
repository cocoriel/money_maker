# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import pymongo
import numpy as np
import statsmodels.tsa.stattools as ts
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC

from datetime import datetime, timedelta, date
import getData
import stuff



### p.133 time_lags인자는 현재일 기준으로 몇일전의 데이터르 ㄹ이요할 것인가를 지정
##        time_lags=5라면 5일 전의 데이터로 예측한다.
def make_dataset(df, include_columns, time_lags=5):
    df_lag = pd.DataFrame(index = df.index)
    #df_lag.index = list(df.Date)
    df_lag["Date"] = df["Date"]
    ## 종가
    df_lag["Close"] = df["Close"]
    ## 거래량
    df_lag["Volume"] = df["Volume"]
    ## MV5, 10, 20
    df_lag[include_columns] = df[include_columns]

    ## shift(5) : 5 row 뒤로 밀림
    df_lag["Close_Lag%s" % str(time_lags)] = df["Close"].shift(time_lags)
    ## pct_change() : 주어진 데이터의 변화를 퍼센트로 계산하는 함수
    df_lag["Close_Lag%s_Change" % str(time_lags)] = df_lag["Close_Lag%s" % str(time_lags)].pct_change()*100.0

    df_lag["Volume_Lag%s" % str(time_lags)] = df["Volume"].shift(time_lags)
    df_lag["Volume_Lag%s_Change" % str(time_lags)] = df_lag["Volume_Lag%s" % str(time_lags)].pct_change()*100.0


    ## 출력변수 : Close_Direction은 주가의 방향을 의미하며 +1, 내렸으면 -1의 값을 가진다. (np.sign(x) : -1 if x < 0 ## 0 if x==0 ## 1 if x > 0)
    df_lag["Close_Direction"] = np.sign(df_lag["Close_Lag%s_Change" % str(time_lags)])
    ## 출력변수 : Volume_Direction은 거래량의 방향을 의미하며 +1, 내렸으면 -1의 값을 가진다.
    df_lag["Volume_Direction"] = np.sign(df_lag["Volume_Lag%s_Change" % str(time_lags)])

    return df_lag.dropna(how='any')

## p.134 사용자가 지정한 입력변수 와 출력변수로 일정 비율로 date를 기준으로 나누어 준다
def split_dataset(df, input_column_array, output_column, split_ratio):
    split_date = get_date_by_percent(df.index[0],df.index[df.shape[0]-1],split_ratio)
    input_data = df[input_column_array]
    output_data = df[output_column]

    X_train = input_data[df.index < split_date]
    X_test = input_data[df.index >= split_date]
    Y_train = output_data[df.index < split_date]
    Y_test = output_data[df.index >= split_date]

    return X_train,X_test,Y_train,Y_test

def get_date_by_percent(start_date,end_date,percent):
    days = (end_date - start_date).days
    target_days = np.trunc(days * percent)
    target_date = start_date + timedelta(days = target_days)

    return target_date

def do_logistic_regression(x_train,y_train):
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    return classifier


def do_random_forest(x_train,y_train):
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    return classifier

def do_svm(x_train,y_train):
    classifier = SVC()
    classifier.fit(x_train, y_train)
    return classifier

def test_classifier(classifier,x_test,y_test):
    pred = classifier.predict(x_test)

    hit_count = 0
    total_count = len(y_test)
    for index in range(total_count):
        if (pred[index]) == (y_test[index]):
            hit_count = hit_count + 1

    hit_ratio = hit_count/total_count
    score = classifier.score(x_test, y_test)
    #print "hit_count=%s, total=%s, hit_ratio = %s" % (hit_count,total_count,hit_ratio)

    return hit_ratio, score
    # Output the hit-rate and the confusion matrix for each model

    #print("%s\n" % confusion_matrix(pred, y_test))


def main():
    stock_code = getData.mongo().get_stock_code()
    print("Stock Code :: len=%d" %(len(stock_code)))

    #### for
    code = "000720.KS" #stock_code[0]  ## for test
    company = stock_code[code.split(".")[0] == stock_code._id].company
    print("code:%s" %(code))

    price = getData.mongo().get_stock_price_by_code(code)
    price = stuff.utils().cal_moving_average(price)

    time_lags=5

    for time_lags in [5, 10, 20, 60, 120, 200]:
        print("\n\n- Time Lags=%s" % (time_lags))
        include_columns = [col for col in price.columns if 'Norm' in col]

        df_dataset = make_dataset(price, include_columns, time_lags=time_lags)
        df_dataset.index = df_dataset.Date.values; del df_dataset["Date"]
        ## 종가만 넣은 경우
        #X_train,X_test,Y_train,Y_test = split_dataset(df_dataset,["Close_Lag%s"%(time_lags)],"Close_Direction",0.75)
        ## 종가 + 거래량 넣은 경우
        X_train,X_test,Y_train,Y_test = split_dataset(df_dataset,\
                                                      ["Close_Lag%s"%(time_lags), "Volume_Lag%s"%(time_lags)] + include_columns,
                                                      "Close_Direction",
                                                      0.75)

        lr_classifier = do_logistic_regression(X_train,Y_train)
        lr_hit_ratio, lr_score = test_classifier(lr_classifier,X_test,Y_test)
        rf_classifier = do_random_forest(X_train,Y_train)
        rf_hit_ratio, rf_score = test_classifier(rf_classifier,X_test,Y_test)
        svm_classifier = do_svm(X_train,Y_train)
        svm_hit_ratio, svm_score = test_classifier(svm_classifier,X_test,Y_test)

        # 적중률
        print("%s" %(company))
        print("%s : Hit Ratio - Logistic Regreesion=%0.3f, RandomForest=%0.3f, SVM=%0.3f" % (code, lr_hit_ratio,rf_hit_ratio,svm_hit_ratio))

