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

st.sidebar.info(role)

if st.sidebar.button("Deploy"):
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

    st.write("Deploying model...")
    # deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1, # number of instances
        instance_type='ml.g4dn.xlarge', #'ml.p3.2xlarge' # ec2 instance type
        endpoint_name='sm-endpoint-gpt-j-6b'
    )

    st.write("Model deployed!")

st.sidebar.caption("Deploy the model to AMazon SageMaker Inference")

if st.sidebar.button("Delete"):
    # delete endpoint
    sm_client = boto3.client('sagemaker')
    sm_client.delete_endpoint(EndpointName='sm-endpoint-gpt-j-6b')
    sm_client.delete_endpoint_config(EndpointConfigName='sm-endpoint-gpt-j-6b')
    sm_client.delete_model(ModelName='sm-endpoint-gpt-j-6b')
    
    st.write("Endpoint deleted!")

st.sidebar.caption("Delete the model from Amazon SageMaker Inference")

if st.sidebar.button("Info"):
    # get endpoint info
    sm_client = boto3.client('sagemaker')
    response = sm_client.describe_endpoint(EndpointName='sm-endpoint-gpt-j-6b')
    st.write(response)

endpoint_name = "sm-endpoint-gpt-j-6b"

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
min_length = st.sidebar.slider("Min length", 0, 100, 100, 10)
max_length = st.sidebar.slider("Max length", 0, 1000, 200, 10)

def generate_text(prompt):
    
    #sagemaker_runtime = boto3.Session(profile_name="default").client('sagemaker-runtime')
    sagemaker_runtime = sm_client

    payload = {"inputs": prompt,'parameters': {
        'min_length': min_length,
        'max_length': max_length,
        'temperature': temp,
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

st.title("My secret little helper")
st.caption("Developed by Tomas Gonzalez.")
prompt = st.text_area("What would you like me to do?")

if st.button("Do it!"):
    generated_text = generate_text(prompt)
    st.write(generated_text)