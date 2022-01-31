import logging
import json
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print('Loading Lambda function')

runtime = boto3.Session().client('sagemaker-runtime')


def lambda_handler(event, context):
    endpoint_name = event['endpoint'] if 'endpoint' in event else 'pytorch-inference-2022-01-29-02-47-47-895'

    print('Context:::', context)
    print('EventType::', type(event))

    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType="application/json",
                                       Accept='application/json',
                                       Body=json.dumps(event))

    result_str = response['Body'].read().decode('utf-8')
    result = json.loads(result_str)

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'},
        'type-result': str(type(result_str)),
        'COntent-Type-In': str(context),
        'body': json.dumps(result)
    }
