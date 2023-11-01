import argparse
import dataclasses
import json
import os
import subprocess
import weaviate

import requests
import time

from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential

from tqdm import tqdm

from data_utils import chunk_directory
""" 
for vectorizer
"resourceName": "oai-poc-dev"
"deploymentId": "ada2"
for generative-openai
{"resourceName": "oai-poc-dev", "deploymentId": "gpt35turbo16k"}
"""



def create_weaviate_class(class_name, vectorizer = "text2vec-openai", resource_name=None, vec_deploymentId=None, gen_deploymentId=None):
    # define the weaviate class to hold the index
    # a unique id for each item is created automatically
    class_obj = {
        "class": class_name,
        "description": "Name of class",
        "vectorizer": vectorizer,
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "moduleConfig": {
            vectorizer: {"resourceName": resource_name, "deploymentId": vec_deploymentId},
            "generative-openai": {"resourceName": resource_name, "deploymentId": gen_deploymentId},
            "vectorizeClassName": False,
        },
        "properties": [
            {
                "dataType": ["text"],
                "description": "Title",
                "name": "title",
            },
            {
                "dataType": ["text"],
                "description": "Url",
                "name": "url",
                "tokenization": "field",
                "moduleConfig": {
                    "text2vec-openai": {
                        "resourceName": resource_name,
                        "deploymentId": vec_deploymentId,
                        "skip": True,
                    }
                }
            },
            {
                "dataType": ["text"],
                "description": "Text content",
                "name": "content",
                "moduleConfig": {
                    "text2vec-openai": {
                        "resourceName": resource_name,
                        "deploymentId": vec_deploymentId,
                        "skip": False,
                    }
                }
            },
            {
                "dataType": ["text"],
                "description": "metadata",
                "name": "metadata",
                "tokenization": "field",
                "moduleConfig": {
                    "text2vec-openai": {
                        "resourceName": resource_name,
                        "deploymentId": vec_deploymentId,
                        "skip": True,
                    }
                }    
            }
        ],
    }
    return class_obj

def create_weaviate_client(url, api_key, additional_headers = None):
    client = weaviate.Client(url=url, auth_client_secret = weaviate.AuthApiKey(api_key=api_key), additional_headers=additional_headers)
    if client.is_ready():
        return client
    else:
        print('The Weaviate database is not active, or the connection to it is not defined correctly')

def add_weaviate_class_to_db(client, class_obj):
    try:
        client.schema.create_class(class_obj)
    except:
        print('The schema could not be created.')


def add_data(client, class_name, data):
    client.batch.configure(batch_size=100)

    with client.batch as batch:
        # Batch import all data
        for i, d in enumerate(data):
        # print(f"importing doc: {i+1}")

            properties = {
                "title": d["title"],
                "url": d["url"],
                "content": d["content"],
                "metadata": d["metadata"],
            }
            client.batch.add_data_object(
                properties,
                class_name,
                )
    count = 0
    for i in range(0, len(data), 100):
        add_data(data[i:i+100])
        count += len(data[i:i+100])
        time.sleep(10)

    return count

def check_if_class_exists(client, class_name):
    return client.schema.exists(class_name=class_name)

def validate_object_count(client, class_name, num_chunks):
    num_objects = client.query.aggregate(class_name).with_meta_count().do()
    if num_objects == num_chunks:
        print(f'Weaviate index {class_name} created with {num_objects}.')
    else:
        print(f'The number of objects in the {class_name} index does not match the number of chunks created during parsing')

def create_index(config, credential, form_recognizer_client=None, embedding_model_endpoint=None, use_layout=False, njobs=4):
    service_name = config["search_service_name"]
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    location = config["location"]
    index_name = config["index_name"]
    language = config.get("language", None)

    if language and language not in SUPPORTED_LANGUAGE_CODES:
        raise Exception(f"ERROR: Ingestion does not support {language} documents. "
                        f"Please use one of {SUPPORTED_LANGUAGE_CODES}."
                        f"Language is set as two letter code for e.g. 'en' for English."
                        f"If you donot want to set a language just remove this prompt config or set as None")


    # check if search service exists, create if not
    if check_if_search_service_exists(service_name, subscription_id, resource_group, credential):
        print(f"Using existing search service {service_name}")
    else:
        print(f"Creating search service {service_name}")
        create_search_service(service_name, subscription_id, resource_group, location, credential=credential)

    # create or update search index with compatible schema
    if not create_or_update_search_index(service_name, subscription_id, resource_group, index_name, config["semantic_config_name"], credential, language, vector_config_name=config.get("vector_config_name", None)):
        raise Exception(f"Failed to create or update index {index_name}")
    
    # chunk directory
    print("Chunking directory...")
    add_embeddings = False
    if config.get("vector_config_name") and embedding_model_endpoint:
        add_embeddings = True
    result = chunk_directory(config["data_path"], num_tokens=config["chunk_size"], token_overlap=config.get("token_overlap",0),
                             azure_credential=credential, form_recognizer_client=form_recognizer_client, use_layout=use_layout, njobs=njobs,
                             add_embeddings=add_embeddings, embedding_endpoint=embedding_model_endpoint)

    if len(result.chunks) == 0:
        raise Exception("No chunks found. Please check the data path and chunk size.")

    print(f"Processed {result.total_files} files")
    print(f"Unsupported formats: {result.num_unsupported_format_files} files")
    print(f"Files with errors: {result.num_files_with_errors} files")
    print(f"Found {len(result.chunks)} chunks")

    # upload documents to index
    print("Uploading documents to index...")
    upload_documents_to_index(service_name, subscription_id, resource_group, index_name, result.chunks, credential)

    # check if index is ready/validate index
    print("Validating index...")
    validate_index(service_name, subscription_id, resource_group, index_name)
    print("Index validation completed")



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file containing settings for data preparation")
    parser.add_argument("--form-rec-resource", type=str, help="Name of your Form Recognizer resource to use for PDF cracking.")
    parser.add_argument("--form-rec-key", type=str, help="Key for your Form Recognizer resource to use for PDF cracking.")
    parser.add_argument("--form-rec-use-layout", default=False, action='store_true', help="Whether to use Layout model for PDF cracking, if False will use Read model.")
    parser.add_argument("--njobs", type=valid_range, default=4, help="Number of jobs to run (between 1 and 32). Default=4")
    parser.add_argument("--embedding-model-endpoint", type=str, help="Endpoint for the embedding model to use for vector search. Format: 'https://<AOAI resource name>.openai.azure.com/openai/deployments/<Ada deployment name>/embeddings?api-version=2023-03-15-preview'")
    parser.add_argument("--embedding-model-key", type=str, help="Key for the embedding model to use for vector search.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    credential = AzureCliCredential()
    form_recognizer_client = None

    print("Data preparation script started")
    if args.form_rec_resource and args.form_rec_key:
        os.environ["FORM_RECOGNIZER_ENDPOINT"] = f"https://{args.form_rec_resource}.cognitiveservices.azure.com/"
        os.environ["FORM_RECOGNIZER_KEY"] = args.form_rec_key
        if args.njobs==1:
            form_recognizer_client = DocumentAnalysisClient(endpoint=f"https://{args.form_rec_resource}.cognitiveservices.azure.com/", credential=AzureKeyCredential(args.form_rec_key))
        print(f"Using Form Recognizer resource {args.form_rec_resource} for PDF cracking, with the {'Layout' if args.form_rec_use_layout else 'Read'} model.")
    else:
        print('Using pypdf to crack PDFs, if any')

    for index_config in config:
        print("Preparing data for index:", index_config["index_name"])
        if index_config.get("vector_config_name") and not args.embedding_model_endpoint:
            raise Exception("ERROR: You must supply an embedding model and key to use Weaviate. Please provide these values or don't use Weaviate.")
    
        create_index(index_config, credential, form_recognizer_client, embedding_model_endpoint=args.embedding_model_endpoint, use_layout=args.form_rec_use_layout, njobs=args.njobs)
        print("Data preparation for index", index_config["index_name"], "completed")

    print(f"Data preparation script completed. {len(config)} indexes updated.")



   
