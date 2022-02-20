import json
import boto3


def get_s3_file_content(key):
    s3 = boto3.resource('s3')
    bucket = 'edgarin-prj-stock-prediction-uw2'  # TODO: Put this in env variable
    csv_object = s3.Object(bucket, key)
    return csv_object.get()['Body'].read().decode('utf-8')


def read_results(training_job_name):
    prefix = f'results/{training_job_name}/results'
    predictions = get_s3_file_content(f'{prefix}_test_predictions.csv')
    fit_summary = get_s3_file_content(f'{prefix}_fit_summary.txt')
    leaderboard = get_s3_file_content(f'{prefix}_leaderboard.csv')
    model_performance = get_s3_file_content(f'{prefix}_model_performance.txt')
    return predictions, fit_summary, leaderboard, model_performance


def lambda_handler(event, context):
    try:
        training_job_name = event['queryStringParameters']['trainingJobName']
        predictions, fit_summary, leaderboard, model_performance = read_results(training_job_name)
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                "trainingJobName": training_job_name,
                "predictions": predictions,
                "fit_summary": fit_summary,
                "leaderboard": leaderboard,
                "model_performance": model_performance
            })
        }
    except Exception as exc:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                "error": repr(exc),
                "event": event
            })
        }

