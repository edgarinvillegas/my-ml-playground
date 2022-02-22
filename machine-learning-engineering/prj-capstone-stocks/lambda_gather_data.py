import json
import time
import datetime
from dateutil.relativedelta import relativedelta
import urllib3
import os
import boto3


def get_data_bin(ticker, period1, period2, suffix=''):
    int_period1 = int(time.mktime(period1.timetuple()))
    int_period2 = int(time.mktime(period2.timetuple()))
    interval = '1d'  # 1d, 1m
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={int_period1}&period2={int_period2}&interval={interval}&events=history&includeAdjustedClose=true'
    http = urllib3.PoolManager()
    resp = http.request("GET", url)
    bin_content = resp.data.replace(b'Adj Close', b'Adj_Close')
    return bin_content


def upload(ticker, forecast_months, lookback_months, bin_train, bin_test, skip_upload=False):
    str_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    filename_prefix = f'{ticker}-f{forecast_months}-b{lookback_months}-{str_time}'
    # filename_prefix = f'{ticker}_{period3.strftime("%Y-%m-%d")}_f{forecast_months}_b{lookback_months}'
    if skip_upload: return filename_prefix

    train_filename = f'{filename_prefix}_train.csv'
    test_filename = f'{filename_prefix}_test.csv'

    s3 = boto3.client('s3')
    bucket = 'edgarin-prj-stock-prediction-uw2'

    # Method 2: Client.put_object()
    s3.put_object(Body=bin_train, Bucket=bucket, Key=f'data/{train_filename}')
    s3.put_object(Body=bin_test, Bucket=bucket, Key=f'data/{test_filename}')

    return filename_prefix


def gather_data(ticker, forecast_months, lookback_months, skip_upload=False, limit_date=None):
    # Let's calculate 3 milestones. Today, the forecast date (for example a month back) and the lookback date (for example 3 months back)
    period3 = limit_date if limit_date is not None else datetime.date.today()  # Use datetime.datetime(2020, 12, 1, 23, 59) for a specific day

    period2 = period3 - relativedelta(
        months=forecast_months)  # Alternative: datetime.timedelta(days=forecast_months * 30)
    period1 = period2 - relativedelta(
        months=lookback_months)  # Alternative: datetime.timedelta(days=lookback_months * 30)

    bin_train = get_data_bin(ticker, period1, period2)
    bin_test = get_data_bin(ticker, period2, period3)

    training_job_name = upload(ticker, forecast_months, lookback_months, bin_train, bin_test, skip_upload)
    train_csv = bin_train.decode('utf-8')
    test_csv = bin_test.decode('utf-8')

    return (training_job_name, train_csv, test_csv)


def lambda_handler(event, context):
    '''
       event = {
          "queryStringParameters": {
            "ticker": "GLD",
            "forecastMonths": "2",
            "lookbackMonths": "6"
          }
        }

    '''
    try:
        params = event['queryStringParameters']
        skip_upload = True if 'skipUpload' in params and params['skipUpload'] == '1' else False
        ticker = params['ticker']
        forecast_months = int(params['forecastMonths'])
        lookback_months = int(params['lookbackMonths'])

        training_job_name, train_csv, test_csv = gather_data(ticker, forecast_months, lookback_months, skip_upload)

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-type': 'application/json'
            },
            'body': json.dumps({
                "skipUpload": skip_upload,
                "trainingJobName": training_job_name,
                # "trainCsv": train_csv, # Commented because this can be big.
                "testCsv": test_csv,
            })
        }
    except Exception as exc:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-type': 'application/json'
            },
            'body': json.dumps({
                "error": repr(exc),
                "event": event
            })
        }