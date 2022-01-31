# Operationalizing a Machine Learning Project

## Notebook instance creation

We chose a ml.t2.medium instance type. This is because the actual notebook performs light operations and we don't need more power. 
The delegated instances perform the heavy workload, but we'll see that later. 

![](assets/endpoint-setup-01-settings.png)

Then, we create a new role. We only need the notebook to access to the `edgarin-mlend` bucket, so we configure the new role that way:

![](assets/endpoint-setup-02-role.png)

Next step is to configure the other security settings. 
This includes
- Selecting the newly created role
- Disable root access to the notebook
- Select a specific vpc, subnet and security group

![](assets/endpoint-setup-03.png)

We also created a bucket to store the training data and the artifacts (the aforementioned `edgarin-mlend` bucket).

![img.png](assets/bucket.png)

## Training and Endpoint deployment
All the process is performed in the `train-and-deploy.ipynb` notebook. 
In a nutshell, it does the following
- Data setup
- Training and HPO
- Best model deployment

After all this is done, the endpoint is deployed

![img.png](assets/endpoint-deployment.png)

## Training on EC2 instance
The choice is an `ml.m5.xlarge` instance to be able to train in a reasonable amount of time without much cost. 
Given this will be a 1 time run, a spot instance was chosen.

If the code had had CUDA, we would have chosen a `p2.xlarge` instance (cheapest with good GPU).

![img.png](assets/ec2-type.png)

For the security group, we chose to access it only from my ip with a keypair (ssh)
![img.png](assets/ec2-security-group.png)

Then, training is triggered via ssh
![img.png](assets/ec2-training-ssh.png)
This creates a model artifact in TrainedModels/model.pth

This script (`ec2train1.py`) is essentially the same as the one trained in script mode (`hpo.py`) with the following minor differences:
- `hpo.py` does have logging
- `hpo.py` is parameterized by command line or env vars (learning rate, batch size, file locations), whereas `ec2train1.py` has them hardcoded
In fact, we could use `hpo.py` for both purposes, as long as we send the params when calling it.

## Endpoint invocation from Lambda
We create a lambda function that will basically take a dog image URL and will invoke the deployed endpoint to perform the prediction.
It uses boto3's `invoke_endpoint` api, sending a json with the image url.
Let's keep in mind that behind the scenes, the `inference2.py` script will transform this url to a binary image that the endpoint can process.

Finally, the prediction -as defined in the training script- is an array of log probabilities for all breeds:

![img.png](assets/lambda-inference.png)

### About Lambda's security
Before executing the lambda, function, its execution role needs permisions to invoke the endpoint. So, a role with AWSSagemakerFullAccess policy was created:

![img.png](assets/lambda-role-creation.png)

And then of course the role is attached to the lambda function.
![img.png](assets/lambda-role-attachment.png)

In a real life scenario, it's strongly suggested to apply the least privilege principle. According to this, we would create a role with only permissions to invoke the lambda, but not full sagemaker access.

On the other hand, it's very important to configure resource based policies to define who specifically can execute the lambda function.
For instance, if it's through API gateway, restrict the permissions to this service (and also configuring the correct authentication/authorization permissions in the api itself, but that's another story). 

Finally, adding the lambda to a VPC needs to be considered, if applicable. 


