# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback
import datetime
import flask
import boto3
import botocore
from config import APP_CONFIG
import globals
import filecmp
from shutil import copyfile
from joblib import load

import pandas as pd
import numpy as np

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

rel_crds = [803,801,507,115,548,802]
static_dt = pd.DataFrame([[int(val==i) for i in rel_crds] for val in rel_crds])
static_dt.columns = ['cardid_'+str(i) for i in rel_crds]
static_dt['ID'] = rel_crds

median_imp = {'age_avg_installment_loan':22.75,'age_newest_bank_card':3,'age_of_oldest_account':145,'age_of_youngest_open_account':2,'available_credit_limit_all_bank_cards':5280,'avg_age_bank_card':21.18,'avg_age_of_open_accounts':27,'balance_all_delinq_accounts':0,'balance_open_accounts_with_past_delinquencies':0,'balance_student_loans':0,'credit_limit_most_recent_credit_card_capital_one':1000,'credit_limit_most_recent_credit_card_chase':5100,'highest_balance_all_bank_cards':2831,'highest_credit_card_limit':7500,'highest_credit_limit_all_bank_cards':7500,'highest_delinquency_installment_loans':1,'highest_delinquency_real_estate_accounts':1,'lowest_credit_card_limit':300,'overall_balance':5296,'overall_credit_limit':16800,'ratio_balance_to_loan_all_delinq_accounts':0.9699,'ratio_balance_to_loan_open_installment_loans_with_no_past_delinq':0.88,'ratio_of_bank_cards_with_balance':0.7099,'total_amount_owed_on_collection_accounts':0,'total_minimum_monthly_payment_all_accounts':828,'total_minimum_monthly_payment_all_installment_tradelines':406.5,'total_minimum_monthly_payment_all_revolving_tradelines':181,'total_past_delinquencies':0,'vantage_score':667,'has_discharged_bankruptcy':0,'num_bank_cards_opened_last_12months':0,'num_bank_cards_with_credit_limit_above_5000':0,'num_bank_cards_with_current_or_past_delinquency':0,'num_charged_off_credit_cards':0,'num_consolidated_inquiries_last_year':0,'num_revolving_tradelines':0,'num_tradelines':0,'num_tradelines_opened_last_6months':0,'num_unique_open_revolving_tradelines':0,'inquiry_added_timeline_event_in_last_30_days':0,'tradeline_added_timeline_event_in_last_30_days':0}
summ_vars = ['age_avg_installment_loan','age_newest_bank_card','age_of_oldest_account','age_of_youngest_open_account','available_credit_limit_all_bank_cards','avg_age_bank_card','avg_age_of_open_accounts','balance_all_delinq_accounts','balance_open_accounts_with_past_delinquencies','balance_student_loans','credit_limit_most_recent_credit_card_capital_one','credit_limit_most_recent_credit_card_chase','highest_balance_all_bank_cards','highest_credit_card_limit','highest_credit_limit_all_bank_cards','highest_delinquency_installment_loans','highest_delinquency_real_estate_accounts','lowest_credit_card_limit','overall_balance','overall_credit_limit','ratio_balance_to_loan_all_delinq_accounts','ratio_balance_to_loan_open_installment_loans_with_no_past_delinq','ratio_of_bank_cards_with_balance','total_amount_owed_on_collection_accounts','total_minimum_monthly_payment_all_accounts','total_minimum_monthly_payment_all_installment_tradelines','total_minimum_monthly_payment_all_revolving_tradelines','total_past_delinquencies','vantage_score','has_discharged_bankruptcy','num_bank_cards_opened_last_12months','num_bank_cards_with_credit_limit_above_5000','num_bank_cards_with_current_or_past_delinquency','num_charged_off_credit_cards','num_consolidated_inquiries_last_year','num_revolving_tradelines','num_tradelines','num_tradelines_opened_last_6months','num_unique_open_revolving_tradelines']
tmln_vars = ['inquiry_added_timeline_event_in_last_30_days','tradeline_added_timeline_event_in_last_30_days']
feat_list = ['num_bank_cards_with_credit_limit_above_5000','vantage_score','num_charged_off_credit_cards','age_of_oldest_account','highest_balance_all_bank_cards','total_amount_owed_on_collection_accounts','available_credit_limit_all_bank_cards','num_revolving_tradelines','total_minimum_monthly_payment_all_accounts','num_tradelines','num_bank_cards_with_current_or_past_delinquency','lowest_credit_card_limit','highest_credit_limit_all_bank_cards','num_bank_cards_opened_last_12months','credit_limit_most_recent_credit_card_capital_one','total_past_delinquencies','overall_credit_limit','total_minimum_monthly_payment_all_installment_tradelines','ratio_balance_to_loan_open_installment_loans_with_no_past_delinq','age_avg_installment_loan','avg_age_of_open_accounts','overall_balance','highest_delinquency_installment_loans','inquiry_added_timeline_event_in_last_30_days','has_discharged_bankruptcy','age_newest_bank_card','tradeline_added_timeline_event_in_last_30_days','total_minimum_monthly_payment_all_revolving_tradelines','num_consolidated_inquiries_last_year','ratio_balance_to_loan_all_delinq_accounts','balance_student_loans','avg_age_bank_card','num_tradelines_opened_last_6months','highest_credit_card_limit','ratio_of_bank_cards_with_balance','balance_all_delinq_accounts','credit_limit_most_recent_credit_card_chase','age_of_youngest_open_account','highest_delinquency_real_estate_accounts','num_unique_open_revolving_tradelines','balance_open_accounts_with_past_delinquencies','cardid_803','cardid_801','cardid_507','cardid_115','cardid_548','cardid_802']
feat_list_log = ['vantage_score','highest_balance_all_bank_cards','total_amount_owed_on_collection_accounts','available_credit_limit_all_bank_cards','total_minimum_monthly_payment_all_accounts','lowest_credit_card_limit','highest_credit_limit_all_bank_cards','credit_limit_most_recent_credit_card_capital_one','overall_credit_limit','total_minimum_monthly_payment_all_installment_tradelines','overall_balance','total_minimum_monthly_payment_all_revolving_tradelines','balance_student_loans','highest_credit_card_limit','balance_all_delinq_accounts','credit_limit_most_recent_credit_card_chase']
jsonkeys = ['summary_vars','engagment_vars','model_name']
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    prime_model_gbm = None ; prime_model_log = None ; 
    
    @classmethod
    def get_model1(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""      
        if cls.prime_model_gbm == None:
            cls.prime_model_gbm = load(open(os.path.join(model_path, 'nonprime_prime_gbm_mdl.pkl'),'rb'))
        return cls.prime_model_gbm
    @classmethod
    def get_model2(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""      
        if cls.prime_model_log == None:
            cls.prime_model_log = load(open(os.path.join(model_path, 'nonprime_prime_mdl_log.pkl'),'rb'))
        return cls.prime_model_log

    
    @classmethod
    def predict(cls, input):
        mdl_gbm= cls.get_model1()
        mdl_log= cls.get_model2()
        ver = 'DISCOVER_PRIME_USER_MODEL_1.1'
        clndata = create_model_features(input,static_dt)
        if clndata['status']=='fail':
            return {'result':None,'status':'fail','remark':'failed at data cleaning - missing variables'}
        try: 
            pred = get_predictions(clndata['data'],mdl_gbm,mdl_log)
            return {'result':pred[['ID','SCORE']],'status':'pass','remark':'pass','version':ver}
        except:
            return {'result':None,'status':'fail','remark':'failed at scoring','version':ver}

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health1 = ScoringService.get_model1() is not None  # You can insert a health check here
    ping_response = "Docker for Discover non prime users"
    status = 200 if (health1) else 404
    return flask.Response(response=ping_response, status=status, mimetype='application/json')

@app.route('/updateccdata', methods=['GET'])
def updateccdata():
    messages = 'No external files to be pulled'
    return flask.Response(response=messages, status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None
    # Read the JSON input 
    if flask.request.content_type == 'application/json':
        data = flask.request.get_json()
    else:
        return flask.Response(response='{"status":"failed","data":"","error":"API only supports JSON","msg":""}', status=200, mimetype='text/plain')
    
    print('Invoked json with {} elements'.format(len(data)))
    miskeys = [var for var in jsonkeys if var not in data.keys()]
    if len(miskeys)>0 : return flask.Response(response='{"status":"failed","data":"","error":"input json have missing keys - '+','.join(miskeys)+'","msg":""}', status=200, mimetype='text/plain')
    
    if data['model_name'] not in ['DISCOVER_PRIME_USER_MODEL_1.1'] : return flask.Response(response='{"status":"failed","data":"","error":"incorrect model selection","msg":""}', status=200, mimetype='text/plain')
    
    # Do the prediction 
    output = ScoringService.predict(data)
    if output['status']=='fail':
        return flask.Response(response='{"status":"failed","data":[],"error":"'+output['remark']+'","msg":""}', status=200, mimetype='text/plain')
    
    predictions = output['result']       
    resp_json = '{"status":"success","data":'+predictions.to_json(orient='records')+',"error":"","msg":"'+output['status']+'","version":"'+output['version']+'"}'
    return flask.Response(response=resp_json, status=200, mimetype='text/csv')

def create_model_features(dt,static_dt):
    try:
        remark = ''
        static_dt = static_dt.reset_index(drop=True)
        raw_vars = {i: convert_float(dt['summary_vars'][i]) for i in summ_vars}
        raw_vars['inquiry_added_timeline_event_in_last_30_days'] = convert_float(dt['engagment_vars']['inquiry_added_timeline_event_in_last_30_days'])
        raw_vars['tradeline_added_timeline_event_in_last_30_days'] = convert_float(dt['engagment_vars']['tradeline_added_timeline_event_in_last_30_days'])
        vars_list = [imp_value(raw_vars[k],median_imp[k]) for k in summ_vars+tmln_vars]
        vars_summary = pd.DataFrame([vars_list for i in range(static_dt.shape[0])])
        vars_summary.columns = summ_vars+tmln_vars
        cln_data = pd.concat([static_dt,vars_summary],axis=1)

        return {'data':cln_data,'status':'pass','remark':remark}
    except:
        return {'data':None,'status':'fail','remark':'failed at data processing'}   
    
def convert_float(x):
    try:
        return np.float(x)
    except:
        return np.float('nan')
    
def imp_value(val,imp_value):
    return imp_value if (np.isnan(val))  else val

def get_predictions(data,model_gbm,model_log):
    pred = np.around(model_gbm.predict_proba(data[feat_list])[:,1],3)
    ret = pd.concat([data.ID,pd.Series(pred)],axis=1)
    ret.columns = ['ID','SCORE']
    pred = np.round(model_log.predict_proba(data[feat_list_log])[:,1][0],3)
    ret['SCORE'] = np.where(ret.ID.isin([115,548,802]),pred,ret.SCORE)
    return ret
