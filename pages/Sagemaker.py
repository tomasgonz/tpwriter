import streamlit as st
from sagemaker.huggingface import HuggingFaceModel
import sagemaker
import boto3
import json

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SageMakerFullAccess')['Role']['Arn']

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
    sagemaker_runtime = boto3.Session(profile_name="default").client('sagemaker-runtime')
    payload = {"inputs": prompt}
    sagemaker_runtime.invoke_endpoint(
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