import streamlit as st
from sagemaker.huggingface import HuggingFaceModel
import sagemaker
import boto3
import json

sess = sagemaker.Session()

sagemaker_session_bucket=None

if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()


try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SageMakerFullAccess')['Role']['Arn']

    
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
region = sess.boto_region_name
sm_client = boto3.client('sagemaker-runtime')

st.write(role)

if st.button("Deploy model"):
    # public S3 URI to gpt-j artifact
    model_uri="s3://huggingface-sagemaker-models/transformers/4.12.3/pytorch/1.9.1/gpt-j/model.tar.gz"
    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        model_data=model_uri,
        transformers_version='4.12.3',
        pytorch_version='1.9.1',
        py_version='py38',
        role=role, 
    )


    # deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1, # number of instances
        instance_type='ml.g4dn.xlarge', #'ml.p3.2xlarge' # ec2 instance type
        endpoint_name='sm-endpoint-gpt-j-6b'
    )


endpoint_name = st.text_area("Enter endpoint name:")

def generate_text(prompt):
    
    #sagemaker_runtime = boto3.Session(profile_name="default").client('sagemaker-runtime')
    sagemaker_runtime = sm_client

    payload = {"inputs": prompt,'parameters': {
        'min_length': 100,
        'max_length': 200,
        'temperature': 0.7,
    }
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
        )
                                
    result = json.loads(response['Body'].read().decode())
    text = result[0]['generated_text']
    return text

st.header("My very own GPT-J Playground")
prompt = st.text_area("Enter your prompt here:")

if st.button("Run"):
    generated_text = generate_text(prompt)
    st.write(generated_text)