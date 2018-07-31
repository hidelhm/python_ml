#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@Time    : 2018/7/23 8:48
@Author  : Hide
@File    : 3up4model_0723.py
@Software: PyCharm
"""
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import OneHotEncoder

from utils.Template import read_and_handle, data_trans, tree_way
import time
import lightgbm as lgb
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

num_attribs = [
    'balance', 'Bil_1X_Dur', 'Bil_3G_Dur', 'charge', 'cwqf',
    'F1X_flux', 'F3G_flux', 'Fav_Inv_Amt', 'fkzs',
    'fz_flag', 'hbbt', 'Hday_Days', 'Hday_Dur',
    'Hday_Flux', 'Home_Dur', 'Home_Flux', 'Inner_Rgn_Amt',
    'Int_Sms_Amt', 'Inter_Rgn_Amt', 'Inv_Amt', 'lower_charge',
    'O_Call_Cnt', 'O_Call_Dstn', 'O_Call_Dur', 'O_Inet_Pp_Sms_Cnt',
    'O_Inner_Rgn_Cnt', 'O_Inner_Rgn_Dur', 'O_Inter_Rgn_Cnt', 'O_Inter_Rgn_Dur', 'O_Onet_Pp_Sms_Cnt',
    'O_Sp_Sms_Cnt', 'O_Tol_Cnt', 'O_Tol_Dstn', 'O_Tol_Dur', 'Off_Dur', 'Off_Flux',
    'Office_Dur', 'Office_Flux', 'Ofr_Id', 'Sp_Sms_Amt',
    'On_Dur', 'On_Flux', 'Owe_Amt', 'Pp_Sms_Amt', 'Prom_Amt', 'Recv_Rate',
    'T_Call_Cnt', 'T_Call_Dstn', 'T_Call_Dur', 'T_Inet_Pp_Sms_Cnt', 'T_Onet_Pp_Sms_Cnt',
    'T_Sp_Sms_Cnt', 'Tdd_Bil_Dur', 'Tdd_Flux', 'Term_Mob_Price',
    'thsc', 'Total_1X_Cnt', 'Total_3G_Cnt', 'Total_Flux', 'Total_Tdd_Cnt',
    'Wday_Days', 'Wday_Dur', 'Wday_Flux', 'ywqf',
    'Cer_Send_Rate', 'Cer_Recv_Rate', 'pro_Inv_Amt', 'pro_o_dur', 'pro_i_dur', 'pro_cdma_nbt', 'pro_flux',
    'balance_last', 'Bil_1X_Dur_last', 'Bil_3G_Dur_last', 'charge_last', 'cwqf_last',
    'F1X_flux_last', 'F3G_flux_last', 'Fav_Inv_Amt_last', 'fkzs_last', 'fz_flag_last',
    'hbbt_last', 'Hday_Days_last', 'Hday_Dur_last', 'Hday_Flux_last', 'Home_Dur_last', 'Home_Flux_last',
    'Inner_Rgn_Amt_last', 'Int_Sms_Amt_last', 'Inter_Rgn_Amt_last', 'Inv_Amt_last', 'lower_charge_last',
    'O_Call_Cnt_last', 'O_Call_Dstn_last', 'O_Call_Dur_last', 'O_Inet_Pp_Sms_Cnt_last', 'O_Inner_Rgn_Cnt_last',
    'O_Inner_Rgn_Dur_last', 'O_Inter_Rgn_Cnt_last', 'O_Inter_Rgn_Dur_last', 'O_Onet_Pp_Sms_Cnt_last',
    'O_Sp_Sms_Cnt_last', 'O_Tol_Cnt_last', 'O_Tol_Dstn_last', 'O_Tol_Dur_last', 'Off_Dur_last',
    'Off_Flux_last', 'Office_Dur_last', 'Office_Flux_last', 'Ofr_Id_last', 'Sp_Sms_Amt_last',
    'On_Dur_last', 'On_Flux_last', 'Owe_Amt_last', 'Pp_Sms_Amt_last', 'Prom_Amt_last',
    'Recv_Rate_last', 'T_Call_Cnt_last', 'T_Call_Dstn_last', 'T_Call_Dur_last', 'T_Inet_Pp_Sms_Cnt_last',
    'T_Onet_Pp_Sms_Cnt_last', 'T_Sp_Sms_Cnt_last', 'Tdd_Bil_Dur_last', 'Tdd_Flux_last',
    'Term_Mob_Price_last', 'thsc_last', 'Total_1X_Cnt_last', 'Total_3G_Cnt_last', 'Total_Flux_last',
    'Total_Tdd_Cnt_last', 'Wday_Days_last', 'Wday_Dur_last', 'Wday_Flux_last', 'ywqf_last',
    'Cer_Send_Rate_last', 'Cer_Recv_Rate_last', 'pro_Inv_Amt_last', 'pro_o_dur_last', 'pro_i_dur_last',
    'pro_cdma_nbt_last', 'pro_flux_last',
    'balance_last2', 'Bil_1X_Dur_last2', 'Bil_3G_Dur_last2', 'charge_last2', 'cwqf_last2',
    'F1X_flux_last2', 'F3G_flux_last2', 'Fav_Inv_Amt_last2', 'fkzs_last2', 'fz_flag_last2',
    'hbbt_last2', 'Hday_Days_last2', 'Hday_Dur_last2', 'Hday_Flux_last2', 'Home_Dur_last2',
    'Home_Flux_last2', 'Inner_Rgn_Amt_last2', 'Int_Sms_Amt_last2', 'Inter_Rgn_Amt_last2', 'Inv_Amt_last2',
    'lower_charge_last2', 'O_Call_Cnt_last2', 'O_Call_Dstn_last2', 'O_Call_Dur_last2',
    'O_Inet_Pp_Sms_Cnt_last2', 'O_Inner_Rgn_Cnt_last2', 'O_Inner_Rgn_Dur_last2', 'O_Inter_Rgn_Cnt_last2',
    'O_Inter_Rgn_Dur_last2', 'O_Onet_Pp_Sms_Cnt_last2', 'O_Sp_Sms_Cnt_last2', 'O_Tol_Cnt_last2',
    'O_Tol_Dstn_last2', 'O_Tol_Dur_last2', 'Off_Dur_last2', 'Off_Flux_last2', 'Office_Dur_last2',
    'Office_Flux_last2', 'Ofr_Id_last2', 'Sp_Sms_Amt_last2', 'On_Dur_last2', 'On_Flux_last2',
    'Owe_Amt_last2', 'Pp_Sms_Amt_last2', 'Prom_Amt_last2', 'Recv_Rate_last2', 'T_Call_Cnt_last2',
    'T_Call_Dstn_last2', 'T_Call_Dur_last2', 'T_Inet_Pp_Sms_Cnt_last2', 'T_Onet_Pp_Sms_Cnt_last2',
    'T_Sp_Sms_Cnt_last2', 'Tdd_Bil_Dur_last2', 'Tdd_Flux_last2', 'Term_Mob_Price_last2', 'thsc_last2',
    'Total_1X_Cnt_last2', 'Total_3G_Cnt_last2', 'Total_Flux_last2', 'Total_Tdd_Cnt_last2', 'Wday_Days_last2',
    'Wday_Dur_last2', 'Wday_Flux_last2', 'ywqf_last2', 'Cer_Send_Rate_last2', 'Cer_Recv_Rate_last2',
    'pro_Inv_Amt_last2', 'pro_o_dur_last2', 'pro_i_dur_last2', 'pro_cdma_nbt_last2', 'pro_flux_last2', ]

cat_attribs = ['yhfl', 'Channel_Type_Name', 'Channel_Type_Name_Lvl1', 'hhmd', 'Card_Type',
               'Std_Merge_Prom_Type_Id', 'Term_Type_Id', 'Accs_Grade', 'black_flag',
               'Exp_Billing_Cycle_Id', 'Gender_Id', 'Innet_Billing_Cycle_Id', 'Latn_Id',
               'Std_Prd_Inst_Stat_Id', 'Strategy_Segment_Id', 'zhrh', 'Cde_Merge_Prom_Name_n', 'Age_subsection',

               'yhfl_last', 'Channel_Type_Name_last', 'Channel_Type_Name_Lvl1_last', 'hhmd_last',
               'Card_Type_last', 'Std_Merge_Prom_Type_Id_last', 'Term_Type_Id_last', 'Accs_Grade_last',
               'black_flag_last', 'Exp_Billing_Cycle_Id_last', 'Gender_Id_last', 'Innet_Billing_Cycle_Id_last',
               'Latn_Id_last', 'Std_Prd_Inst_Stat_Id_last', 'Strategy_Segment_Id_last', 'zhrh_last',
               'Cde_Merge_Prom_Name_n_last', 'Age_subsection_last',
               'yhfl_last2', 'Channel_Type_Name_last2', 'Channel_Type_Name_Lvl1_last2', 'hhmd_last2',
               'Card_Type_last2', 'Std_Merge_Prom_Type_Id_last2', 'Term_Type_Id_last2', 'Accs_Grade_last2',
               'black_flag_last2', 'Exp_Billing_Cycle_Id_last2', 'Gender_Id_last2', 'Innet_Billing_Cycle_Id_last2',
               'Latn_Id_last2', 'Std_Prd_Inst_Stat_Id_last2', 'Strategy_Segment_Id_last2', 'zhrh_last2',
               'Cde_Merge_Prom_Name_n_last2', 'Age_subsection_last2', ]

flag = ['flag']

id = "Prd_Inst_Id"
label = "flag"
time_time = int(time.time() * 1000)


def load_data(filename):
    cols = [
        'Accs_Mode_Cd', 'Accs_Nbr', 'Acct_Id', 'Age',
        'balance', 'Bil_1X_Dur', 'Bil_3G_Dur', 'Billing_Type_Id', 'black_flag',
        'Card_Type', 'Cde_Merge_Prom_Name', 'Cert_Nbr', 'cfq_end_date', 'Channel_Type_Name',
        'Channel_Type_Name_Lvl1', 'charge', 'cwqf', 'dzq', 'End_Date',
        'Exp_Billing_Cycle_Id', 'F1X_flux', 'F3G_flux', 'Fav_Inv_Amt', 'fkzs',
        'fz_flag', 'Gender_Id', 'hbbt', 'Hday_Days', 'Hday_Dur',
        'Hday_Flux', 'hhmd', 'Home_Dur', 'Home_Flux', 'Inner_Rgn_Amt',
        'Innet_Billing_Cycle_Id', 'Int_Sms_Amt', 'Inter_Rgn_Amt', 'Inv_Amt', 'kdll',
        'Latn_Id', 'Latn_Name', 'lower_charge', 'Merge_Prom_Inst_Id', 'Mkt_Emp_name',
        'Name', 'O_Call_Cnt', 'O_Call_Dstn', 'O_Call_Dur', 'O_Inet_Pp_Sms_Cnt',
        'O_Inner_Rgn_Cnt', 'O_Inner_Rgn_Dur', 'O_Inter_Rgn_Cnt', 'O_Inter_Rgn_Dur', 'O_Onet_Pp_Sms_Cnt',
        'O_Sp_Sms_Cnt', 'O_Tol_Cnt', 'O_Tol_Dstn', 'O_Tol_Dur', 'Off_Dur', 'Off_Flux',
        'Office_Dur', 'Office_Flux', 'Ofr_Id', 'Ofr_Inst_Id', 'Ofr_Name',
        'On_Dur', 'On_Flux', 'Owe_Amt', 'Pay_Flag', 'Pp_Sms_Amt',
        'Prd_Id', 'Prd_Inst_Id', 'Prd_Name', 'Prom_Amt', 'Recv_Rate',
        'Sp_Sms_Amt', 'Std_Merge_Prom_Type_Id', 'Std_Prd_Id', 'Std_Prd_Inst_Stat_Id', 'Strategy_Segment_Id',
        'T_Call_Cnt', 'T_Call_Dstn', 'T_Call_Dur', 'T_Inet_Pp_Sms_Cnt', 'T_Onet_Pp_Sms_Cnt',
        'T_Sp_Sms_Cnt', 'Tdd_Bil_Dur', 'Tdd_Flux', 'Term_Mob_Price', 'Term_Type_Id',
        'thsc', 'Total_1X_Cnt', 'Total_3G_Cnt', 'Total_Flux', 'Total_Tdd_Cnt',
        'Wday_Days', 'Wday_Dur', 'Wday_Flux', 'yhfl', 'ywqf', 'zhrh', 'Cde_Merge_Prom_Name_n', 'Age_subsection',
        'Cer_Send_Rate', 'Cer_Recv_Rate', 'pro_Inv_Amt', 'pro_o_dur', 'pro_i_dur', 'pro_cdma_nbt', 'pro_flux',
        'flag'
    ]

    names = num_attribs + cat_attribs + flag

    df = read_and_handle(filename, cat_attribs, names)

    df.to_csv('20180730.csv')

    return df


def single_model_lgb(d_train):
    # d_train = d_train.iloc[:100, :]
    # d_train = data_trans(d_train)

    # d_train = data_trans(d_train)

    X = d_train[[x for x in d_train.columns if x not in [id, label]]]
    y = d_train[label]

    # selector = VarianceThreshold()
    # X2 = selector.fit_transform(X)
    # X2 = pd.DataFrame(X2, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('Start training...')
    gbm = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=50, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=5)

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # print('Feature importances:', list(gbm.feature_importances_))
    # estimator = lgb.LGBMRegressor(num_leaves=31)
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20, 40]
    # }
    # gbm = GridSearchCV(estimator, param_grid)
    # gbm.fit(X_train, y_train)
    # print('Best parameters found by grid search are:', gbm.best_params_)
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)

    plot_learning_curve(gbm)
    lgb_name = 'lgb_{}.pkl'.format(int(time.time() * 1000))
    joblib.dump(gbm, lgb_name)

    # clf = joblib.load('lgb_1532917294446.pkl')


def plot_learning_curve(clf):
    # plt.figure(figsize=(12, 6))
    plt.figure
    lgb.plot_importance(clf, max_num_features=30)
    plt.title("Featurertances")
    plt.show()


if __name__ == '__main__':
    # filename = 'D:\\data\\cdmals_543f_6l.txt'
    # train = load_data(filename)

    filename = 'D:\\workspacepycharm\\pysparkml\\3up4model\\20180730.csv'
    train = pd.read_csv(filename, sep=',', header=0, iterator=True, chunksize=100000)
    train = train.get_chunk(200000)
    stime = time.time()
    single_model_lgb(train)
    print("耗时:{}".format(time.time() - stime))
