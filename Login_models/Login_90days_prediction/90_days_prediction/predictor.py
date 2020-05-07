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
import time
import pandas as pd
import numpy as np
from joblib import load

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

rawvars_engagment = ['days_since_registration','days_since_last_ad_viewed','count_ad_views_in_last_90days', 'count_ad_views_in_last_60days','count_ad_views_in_last_30days','count_ad_views_in_last_15days', 'count_credit_cards_page_ad_views_in_last_90days','count_credit_cards_page_ad_views_in_last_60days', 'count_credit_cards_page_ad_views_in_last_30days','count_credit_cards_page_ad_views_in_last_15days','count_ad_clicks_in_last_90days', 'count_ad_clicks_in_last_60days','count_ad_clicks_in_last_30days','count_ad_clicks_in_last_15days', 'num_of_timeline_email_sent_in_last_90days','num_of_inquiry_alert_email_sent_in_last_90days', 'num_of_timeline_score_decrease_email_sent_in_last_90days','num_of_timeline_score_increase_email_sent_in_last_90days', 'num_of_record_credit_score_email_sent_in_last_90days','num_of_monthly_debt_progress_email_sent_in_last_90days', 'num_of_birthday_email_sent_in_last_90days','num_of_negative_event_removed_email_sent_in_last_90days', 'num_of_credit_improvement_addcreditcard_simulation_email_sent_in_last_90days','num_of_monthly_email_sent_in_last_90days']

rawvars_summvars = ['num_inquiries_last_month','num_inquiries_last_3months','num_open_credit_cards','ratio_balance_to_loan_all_accounts', 'total_amount_owed_on_collection_accounts','overall_balance','overall_credit_limit','overall_utilization','age_of_oldest_account', 'age_of_youngest_open_account','available_credit_limit','num_credit_cards','num_auto_loans','num_mortgages']

selvars = ['days_since_registration',
           'days_since_last_ad_viewed','count_ad_views_in_last_90days','count_ad_views_in_last_60days','count_ad_views_in_last_30days',
       'count_ad_views_in_last_15days','count_credit_cards_page_ad_views_in_last_90days','count_credit_cards_page_ad_views_in_last_60days',
      'count_credit_cards_page_ad_views_in_last_30days','count_credit_cards_page_ad_views_in_last_15days','count_ad_clicks_in_last_90days',
           'count_ad_clicks_in_last_60days','count_ad_clicks_in_last_30days','count_ad_clicks_in_last_15days',
           'num_of_timeline_email_sent_in_last_90days','num_of_inquiry_alert_email_sent_in_last_90days',
           'num_of_timeline_score_decrease_email_sent_in_last_90days','num_of_timeline_score_increase_email_sent_in_last_90days',
           'num_of_record_credit_score_email_sent_in_last_90days', 'num_of_monthly_debt_progress_email_sent_in_last_90days',
           'num_of_birthday_email_sent_in_last_90days','num_of_negative_event_removed_email_sent_in_last_90days',
           'num_of_credit_improvement_addcreditcard_simulation_email_sent_in_last_90days','num_of_monthly_email_sent_in_last_90days',
           'num_inquiries_last_3months','num_open_credit_cards','ratio_balance_to_loan_all_accounts',
           'total_amount_owed_on_collection_accounts','overall_balance','overall_credit_limit','overall_utilization',
           'age_of_oldest_account','age_of_youngest_open_account','available_credit_limit','num_credit_cards','num_auto_loans',
           'num_mortgages','is_inquiry_expected_soon']

jsonkeys = ['summary_vars','engagement_vars','model_name']
list_modelname = ['Login_90day_model_V1']

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    model1 = None ;
    
    @classmethod
    def get_model1(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""      
        if cls.model1 == None:
            cls.model1 = load('/opt/ml/model/finalized_model.sav') 
        return cls.model1
   
    @classmethod
    def predict(cls, input):
        model_1 = cls.get_model1()
        ver = '90day_login_model_1.1'
        remark = ''
        clndata = create_model_features(input)
        if clndata['status']=='fail':
            ls_summary_vars = input['summary_vars'].keys()
            ls_engagement_vars = input['engagement_vars'].keys()
            s = 'missing vars - '
            for i in rawvars_engagment:
                if i not in ls_engagement_vars: s = s+i+','
            for i in rawvars_summvars:
                if i not in ls_summary_vars: s = s+i+','
            return {'result':None,'status':'fail','remark':clndata['remark'],'msg':s}
        data_model = clndata['data']
        try:
            pred = get_prediction(data_model,model_1)
        except:
            return {'data':None,'status':'fail','remark':'failed at model scoring','msg':''}

        return {'result':pred,'status':'pass','remark':clndata['remark']+remark,'version':ver}

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    health = ScoringService.get_model1() is not None  # You can insert a health check here

    ping_response = "Login model 90 days window - {}".format(APP_CONFIG['DEV_ENV'])
    status = 200 if (health) else 404
    return flask.Response(response=ping_response, status=status, mimetype='application/json')

@app.route('/updateccdata', methods=['GET'])
def updateccdata():
    messages = 'No external data'
    messages.append(APP_CONFIG['DEV_ENV'])
    return flask.Response(response=messages, status=200, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    data = None
    # Read the JSON input 
    if flask.request.content_type == 'application/json':
        data = flask.request.get_json()
    else:
        return flask.Response(response='{"status":"failed","data":[],"error":"This predictor only supports JSON","msg":""}', status=200, mimetype='text/plain')
    
    print('Invoked json with {} elements'.format(len(data)))
    miskeys = [var for var in jsonkeys if var not in data.keys()]
    if len(miskeys)>0 : return flask.Response(response='{"status":"failed","data":[],"error":"input json have missing keys - '+','.join(miskeys)+'","msg":""}', status=200, mimetype='text/plain')
    
    if data['model_name'] not in list_modelname : return flask.Response(response='{"status":"failed","data":[],"error":"incorrect model selection","msg":""}', status=200, mimetype='text/plain')
      
    # Do the prediction
    output = ScoringService.predict(data)
    if output['status']=='fail':
        return flask.Response(response='{"status":"failed","data":[],"error":"'+output['remark']+'","msg":"'+output['msg']+'"}', status=200, mimetype='text/plain')
    output_str = str(output['result'])
    resp_json = '{"status":"success","prediction":['+output_str+'],"error":"","msg":"'+output['status']+','+output['remark']+'","version":"'+output['version']+'"}'
    return flask.Response(response=resp_json, status=200, mimetype='text/csv')

def convert_float(x):
    try:
        return np.float(x)
    except:
        return np.float('nan')

def imp_value(val,imp_value):
    return imp_value if (np.isnan(val))  else val

def create_model_features(dt):
    try:
        remark = ''
        dict_vars = {k: convert_float(dt['engagement_vars'][k]) for k in rawvars_engagment}
        dict_add = {k: convert_float(dt['summary_vars'][k]) for k in rawvars_summvars}
        dict_vars.update(dict_add)
        dict_vars['days_since_last_ad_viewed'] = imp_value(dict_vars['days_since_last_ad_viewed'],9999)
        dict_vars['num_inquiries_last_3months'] = imp_value(dict_vars['num_inquiries_last_3months'],999)
        dict_vars['num_open_credit_cards'] = imp_value(dict_vars['num_open_credit_cards'],99)
        dict_vars['ratio_balance_to_loan_all_accounts'] = imp_value(dict_vars['ratio_balance_to_loan_all_accounts'],0.999)
        dict_vars['total_amount_owed_on_collection_accounts'] = imp_value(dict_vars['total_amount_owed_on_collection_accounts'],9999)
        dict_vars['overall_balance'] = imp_value(dict_vars['overall_balance'],9999)
        dict_vars['overall_credit_limit'] = imp_value(dict_vars['overall_credit_limit'],9999)
        dict_vars['overall_utilization'] = imp_value(dict_vars['overall_utilization'],0.999)
        dict_vars['age_of_oldest_account'] = imp_value(dict_vars['age_of_oldest_account'],9999)
        dict_vars['age_of_youngest_open_account'] = imp_value(dict_vars['age_of_youngest_open_account'],9999)
        dict_vars['available_credit_limit'] = imp_value(dict_vars['available_credit_limit'],9999)
        dict_vars['num_credit_cards'] = imp_value(dict_vars['num_credit_cards'],99)
        dict_vars['num_auto_loans'] = imp_value(dict_vars['num_auto_loans'],99)
        dict_vars['num_mortgages'] = imp_value(dict_vars['num_mortgages'],99)
        dict_vars['is_inquiry_expected_soon'] = dict_vars['overall_utilization'] > 0.75 and dict_vars['num_inquiries_last_month'] == 0
        sam_inp = [dict_vars[i] for i in selvars]
        data_model = pd.DataFrame(np.array(sam_inp).reshape(1, -1))
        data_model.columns = selvars
 
        return {'data':data_model,'status':'pass','remark':remark}
    except:
        return {'data':None,'status':'fail','remark':'failed at data processing'}

def get_prediction(model_data,model):
    pred = np.round(model.predict_proba(model_data[selvars])[0,1],3)
    return(pred)