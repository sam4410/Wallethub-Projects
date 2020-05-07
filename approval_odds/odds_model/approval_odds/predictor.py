# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import datetime
import flask
import boto3
import botocore
import time
from config import APP_CONFIG
import globals
import filecmp
from shutil import copyfile
import pandas as pd
import numpy as np
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

summary_vars_all = ['age_of_oldest_account','age_of_youngest_open_account','available_credit_limit','available_credit_limit_all_bank_cards','available_credit_limit_cu','avg_age_of_all_accounts','avg_age_of_open_accounts','avg_days_delinq_last_12months','avg_days_delinq_last_24months','avg_days_delinq_last_36months','avg_days_delinq_last_48months','avg_days_delinq_last_48payments','balance_all_accounts','balance_all_bank_cards','balance_all_delinq_accounts','balance_open_accounts','balance_open_accounts_with_no_past_delinquencies','balance_open_installment_loans','num_consolidated_inquiries_last_3months','num_consolidated_inquiries_last_6months','num_consolidated_inquiries_last_year','num_current_tradelines_dropped','num_inquiries_last_3months','num_inquiries_last_6months','num_inquiries_last_month','num_inquiries_last_year','num_inquries','num_open_credit_cards','num_open_credit_cards_capital_one','num_revolving_tradelines','num_tradelines_above_1000_credit_limit_cu','num_tradelines_opened_last_5year','num_unpaid_collection_accounts_in_last_3years','overall_balance','overall_credit_limit','overall_credit_limit_cu','overall_utilization','ratio_balance_to_loan_all_accounts','total_amount_owed_on_collection_accounts','total_minimum_monthly_payment_all_accounts','total_minimum_monthly_payment_all_auto_accounts','total_minimum_monthly_payment_all_credit_cards','total_minimum_monthly_payment_all_installment_tradelines','vantage_score']
selcrds = [75,76,83,89,99,121,146,378,379,381,382,383,499,503,567,864,942,1524,1667,2084,2122,2286,2293,3067,3202,180,676,905,2294]
imp_dict = {'age_of_oldest_account':100,'age_of_youngest_open_account':3,'available_credit_limit':75,'available_credit_limit_all_bank_cards':20,'available_credit_limit_cu':48,'avg_age_of_all_accounts':20,'avg_age_of_open_accounts':21,'avg_days_delinq_last_12months':62.5,'avg_days_delinq_last_24months':41,'avg_days_delinq_last_36months':26,'avg_days_delinq_last_48months':15,'avg_days_delinq_last_48payments':15,'balance_all_accounts':9749,'balance_all_bank_cards':436,'balance_all_delinq_accounts':0,'balance_open_accounts':4515,'balance_open_accounts_with_no_past_delinquencies':1638,'balance_open_installment_loans':970,'num_consolidated_inquiries_last_3months':1,'num_consolidated_inquiries_last_6months':1,'num_consolidated_inquiries_last_year':2,'num_current_tradelines_dropped':4,'num_inquiries_last_3months':1,'num_inquiries_last_6months':2,'num_inquiries_last_month':0,'num_inquiries_last_year':3,'num_inquries':6,'num_open_credit_cards':1,'num_open_credit_cards_capital_one':0,'num_revolving_tradelines':3,'num_tradelines_above_1000_credit_limit_cu':0,'num_tradelines_opened_last_5year':4,'num_unpaid_collection_accounts_in_last_3years':1,'overall_balance':2,'overall_credit_limit':300,'overall_credit_limit_cu':300,'overall_utilization':0,'ratio_balance_to_loan_all_accounts':0.54,'total_amount_owed_on_collection_accounts':618,'total_minimum_monthly_payment_all_accounts':113,'total_minimum_monthly_payment_all_auto_accounts':0,'total_minimum_monthly_payment_all_credit_cards':0,'total_minimum_monthly_payment_all_installment_tradelines':0,'vantage_score':620}
cardvars = ['capital_one_card','chase_card','citibank_card','card_credit_req_bad','card_credit_req_limited','card_credit_req_fair','card_credit_req_good','card_credit_req_excellent','student_card','personal_card','secured_card','credit_card','rewards_card','cashback_rewrads_card','miles_rewards_card','offers_initial_bonus','offers_intro_purchase_apr','offers_balance_transfer','balance_transfer_fee_none','annual_fee_0']
featurelist = ['vantage_score','num_consolidated_inquiries_last_3months','num_inquiries_last_6months','overall_utilization','num_open_credit_cards_capital_one','avg_age_of_all_accounts','num_inquiries_last_3months','num_open_credit_cards','avg_age_of_open_accounts','num_inquries','num_inquiries_last_month','num_tradelines_opened_last_5year','total_minimum_monthly_payment_all_credit_cards','num_current_tradelines_dropped','total_minimum_monthly_payment_all_installment_tradelines','balance_all_bank_cards','num_revolving_tradelines','ratio_balance_to_loan_all_accounts','total_minimum_monthly_payment_all_auto_accounts','num_unpaid_collection_accounts_in_last_3years','avg_days_delinq_last_36months','total_minimum_monthly_payment_all_accounts','overall_balance','total_amount_owed_on_collection_accounts','balance_open_installment_loans','available_credit_limit_all_bank_cards','available_credit_limit_cu','available_credit_limit','balance_all_accounts','num_tradelines_above_1000_credit_limit_cu','overall_credit_limit','balance_all_delinq_accounts','cardid_75','cardid_76','cardid_83','cardid_99','cardid_121','cardid_378','cardid_379','cardid_381','cardid_382','cardid_383','cardid_499','cardid_503','cardid_567','cardid_864','cardid_942','cardid_1524','cardid_1667','cardid_2122','cardid_2286','cardid_2293','cardid_180','cardid_676','cardid_905','cardid_2294','capital_one_card','chase_card','citibank_card','card_credit_req_bad','card_credit_req_limited','card_credit_req_fair','card_credit_req_good','card_credit_req_excellent','student_card','personal_card','secured_card','credit_card','rewards_card','cashback_rewrads_card','miles_rewards_card','offers_initial_bonus','offers_intro_purchase_apr','offers_balance_transfer','balance_transfer_fee_none','annual_fee_0','diff_overall_available_credit_limit','ratio_avg_age_oldest_account','ratio_num_inquries_last_year','ratio_avg_age_of_open_accounts','ratio_age_of_youngest_open_account','ratio_balance_open_account','ratio_avg_days_delinq_last_48payments','ratio_consolidated_inquiries_last_6months','ratio_avg_days_delinq_last_12months']
summary_vars_model = ['vantage_score','num_consolidated_inquiries_last_3months','num_inquiries_last_6months','overall_utilization','num_open_credit_cards_capital_one','avg_age_of_all_accounts','num_inquiries_last_3months','num_open_credit_cards','avg_age_of_open_accounts','num_inquries','num_inquiries_last_month','num_tradelines_opened_last_5year','total_minimum_monthly_payment_all_credit_cards','num_current_tradelines_dropped','total_minimum_monthly_payment_all_installment_tradelines','balance_all_bank_cards','num_revolving_tradelines','ratio_balance_to_loan_all_accounts','total_minimum_monthly_payment_all_auto_accounts','num_unpaid_collection_accounts_in_last_3years','avg_days_delinq_last_36months','total_minimum_monthly_payment_all_accounts','overall_balance','total_amount_owed_on_collection_accounts','balance_open_installment_loans','available_credit_limit_all_bank_cards','available_credit_limit_cu','available_credit_limit','balance_all_accounts','num_tradelines_above_1000_credit_limit_cu','overall_credit_limit','balance_all_delinq_accounts','diff_overall_available_credit_limit','ratio_avg_age_oldest_account','ratio_num_inquries_last_year','ratio_avg_age_of_open_accounts','ratio_age_of_youngest_open_account','ratio_balance_open_account','ratio_avg_days_delinq_last_48payments','ratio_consolidated_inquiries_last_6months','ratio_avg_days_delinq_last_12months']
jsonkeys = ['summary_vars','model_name','odds_universe']
list_modelname = ['AO_MODEL_V2']
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    model1 = None ;
    
    @classmethod
    def get_model1(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""      
        if cls.model1 == None:
            with open(os.path.join(model_path, 'gbm_approval_application.pkl'), 'r') as inp:
                cls.model1 = pickle.load(inp)
        return cls.model1
   
    @classmethod
    def predict(cls, input):
        md1 = cls.get_model1()
        ver = 'model_1.3'
        remark = ''
        try:
            cardvars_clean = pd.read_csv('/opt/ml/data/card_variables_static.csv')
        except:
            return {'result':None,'status':'fail','remark':'data files not pulled'}
        crds_rel = [int(i) for i in input['odds_universe']]
        cardvars_clean = cardvars_clean[cardvars_clean.ID.isin(crds_rel)]
        if cardvars_clean.shape[0]<1:
            return {'result':None,'status':'fail','remark':'no eligible card from odds_universe'}

        drop_crds = [i for i in crds_rel if i not in cardvars_clean.ID.unique()]
        if len(drop_crds)>0:
            remark = remark+' cards dropped ['+",".join(map(str,drop_crds))+'],'
        clndata = create_model_features(input,cardvars_clean)
        if clndata['status']=='fail':
            return {'result':None,'status':'fail','remark':clndata['remark']}
        
        try: 
            rel_fnl = get_predictions(clndata['data'],md1)
            return {'result':rel_fnl[['ID','APPROVAL_ODDS']],'status':'pass','remark':clndata['remark']+remark,'version':ver}
        except:
            return {'result':None,'status':'fail','remark':'failed at scoring'}

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health1 = ScoringService.get_model1() is not None  # You can insert a health check here

    ping_response = "Hello World from {}".format(APP_CONFIG['DEV_ENV'])
    status = 200 if (health1) else 404
    return flask.Response(response=ping_response, status=status, mimetype='application/json')

@app.route('/updateccdata', methods=['GET'])
def updateccdata():
    messages = []
    messages.append(APP_CONFIG['DEV_ENV'])

    # S3 variables
    env_path = 'dev'
    if ('DEV_ENV' in APP_CONFIG) and APP_CONFIG['DEV_ENV'].find('PROD-ITHACA')>=0:
        env_path = 'prod'
    DATA_S3_PATH = 'data/creditcards/' + env_path + '/'
    METADATA_FILE_S3_KEY = DATA_S3_PATH + globals.METADATA_FILE
    CARD_VARS_FILE_S3_KEY = DATA_S3_PATH + globals.CARD_VARS_FILE

    try:
        session = boto3.Session(
                aws_access_key_id=APP_CONFIG['DATA_S3_ACCESS'],
                aws_secret_access_key=APP_CONFIG['DATA_S3_SECRET'],
        )
        s3client = session.resource('s3')
        s3client.Bucket(APP_CONFIG['DATA_S3_BUCKET']).download_file(METADATA_FILE_S3_KEY, globals.LATEST_METADATA_FILE_LOCAL)
        
        latest_md_file = globals.LATEST_METADATA_FILE_LOCAL
        current_md_file = globals.METADATA_FILE_LOCAL

        if not filecmp.cmp(latest_md_file, current_md_file):
            s3client.Bucket(APP_CONFIG['DATA_S3_BUCKET']).download_file(CARD_VARS_FILE_S3_KEY, globals.CARD_VARS_FILE_LOCAL)
            messages.append("Updated card variables")
            try:
                card_variables = pd.read_csv(globals.CARD_VARS_FILE_LOCAL)
                card_variables_clean = getcardvars(card_variables)
                card_variables_clean.to_csv('/opt/ml/data/card_variables_static.csv',index=None)
                messages.append("Updated card vars clean")
            except:
                messages.append("Issue-with card vars")

            copyfile(latest_md_file, current_md_file)
            messages.append("Updated data, loc -")
            messages.append(env_path)
        else:
            messages.append("No updates!")                    

        messages.insert(0, "Success")

    except Exception as e:
        messages.insert(0, "Error") 
        messages.append(str(e))
    

    msg = ";".join(messages)

    return flask.Response(response=msg, status=200, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
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
      
    try:
        vscore = convert_float(data['summary_vars']['vantage_score'])
        if ((np.isnan(vscore))|(vscore>850)|((vscore<300)&(vscore!=4))) : return flask.Response(response='{"status":"failed","data":[],"error":"incorrect vantage score","msg":""}', status=200, mimetype='text/plain')
    except:
        return flask.Response(response='{"status":"failed","data":[],"error":"incorrect vantage score","msg":""}', status=200, mimetype='text/plain')

    # Do the prediction 
    output = ScoringService.predict(data)
    if output['status']=='fail':
        return flask.Response(response='{"status":"failed","data":[],"error":"'+output['remark']+'","msg":""}', status=200, mimetype='text/plain')
    predictions = output['result']
            
    resp_json = '{"status":"success","data":'+predictions.to_json(orient='records')+',"error":"","msg":"'+output['status']+','+output['remark']+'","version":"'+output['version']+'"}'
    return flask.Response(response=resp_json, status=200, mimetype='text/csv')

def create_model_features(dt,rel_static_dt):
    try:
        remark = ''
        rel_static_dt = rel_static_dt.reset_index(drop=True)
        dict_sumvars = {k: convert_float(v) for k, v in dict(dt['summary_vars']).items()}  
        vars_summ = {k: imp_value(dict_sumvars[k],imp_dict[k]) for k in summary_vars_all}
        vars_summ['diff_overall_available_credit_limit'] = vars_summ['overall_credit_limit_cu'] - vars_summ['available_credit_limit_cu']
        vars_summ['ratio_avg_age_oldest_account'] = vars_summ['avg_age_of_all_accounts']/(np.abs(vars_summ['age_of_oldest_account'])+1)
        vars_summ['ratio_num_inquries_last_year'] = vars_summ['num_inquiries_last_year']/(vars_summ['num_inquries']+1)
        vars_summ['ratio_avg_age_of_open_accounts'] = vars_summ['avg_age_of_open_accounts']/(np.abs(vars_summ['avg_age_of_all_accounts'])+1)
        vars_summ['ratio_age_of_youngest_open_account'] = vars_summ['age_of_youngest_open_account']/(np.abs(vars_summ['avg_age_of_open_accounts'])+1)
        vars_summ['ratio_balance_open_account'] = vars_summ['balance_open_accounts_with_no_past_delinquencies']/(np.abs(vars_summ['balance_open_accounts'])+1)
        vars_summ['ratio_consolidated_inquiries_last_6months'] = vars_summ['num_consolidated_inquiries_last_6months']/(vars_summ['num_consolidated_inquiries_last_year']+1)
        vars_summ['ratio_avg_days_delinq_last_48payments'] = vars_summ['avg_days_delinq_last_48payments']/(np.abs(vars_summ['avg_days_delinq_last_48months'])+1)
        vars_summ['ratio_avg_days_delinq_last_12months'] = vars_summ['avg_days_delinq_last_12months']/(np.abs(vars_summ['avg_days_delinq_last_24months'])+1)

        vars_list = [vars_summ[i] for i in summary_vars_model]
        vars_summary = pd.DataFrame([vars_list for i in range(rel_static_dt.shape[0])])
        vars_summary.columns = summary_vars_model
        cln_data = pd.concat([rel_static_dt,vars_summary],axis=1)

        return {'data':cln_data,'status':'pass','remark':remark}
    except:
        return {'data':None,'status':'fail','remark':'failed at data processing'}

def convert_float(x):
    try:
        return np.float(x)
    except:
        return np.float('nan')

def getcardvars(card_variables):
    colname = card_variables.columns.tolist()
    colname[0] = 'Card_ID'
    card_variables.columns = colname
    tmp = card_variables[~card_variables.Card_ID.isin(selcrds)]
    af_cards = tmp.Card_ID.unique().tolist()+selcrds
    start_dt = pd.DataFrame([[int(val==i) for i in af_cards] for val in af_cards])
    start_dt.columns = ['cardid_'+str(i) for i in af_cards]
    start_dt['ID'] = af_cards
    for i in cardvars:
        lst_crds = card_variables.Card_ID[card_variables[i]].values
        start_dt[i] = np.where(start_dt.ID.isin(lst_crds),1,0)

    return(start_dt)

def convert_float(x):
    try:
        return np.float(x)
    except:
        return np.float('nan')
    
def imp_value(val,imp_value):
    return imp_value if (np.isnan(val))  else val

def get_predictions(data,model):
    pred = model.predict_proba(data[featurelist])[:,1]
    odds = (1.29*pred-0.602*pred**2-0.00692)*1.1
    ret = pd.concat([data.ID,pd.Series(pred),pd.Series(odds)],axis=1)
    ret.columns = ['ID','PREDICTON','APPROVAL_ODDS']
    ret['APPROVAL_ODDS'] = np.where(ret['APPROVAL_ODDS']<0,0.002,np.where(ret['APPROVAL_ODDS']>0.9,0.9,np.around(ret['APPROVAL_ODDS'],3)))
    return ret
