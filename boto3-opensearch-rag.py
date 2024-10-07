import boto3
import json
from opensearchpy import OpenSearch
import os

# Initialize the Boto3 client for AWS Bedrock

bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Initialize the OpenSearch client
opensearch_client = OpenSearch(
    hosts=[{'host': 'your-opensearch-domain', 'port': 443}],
    http_auth=('username', 'password'),  # Replace with your credentials
    use_ssl=True,
    verify_certs=True,
    connection_class='RequestsHttpConnection'
)

# Function to generate embeddings from AWS Bedrock
def generate_embeddings_bedrock(text, model_id="ai21.j2-grande-instruct"):
    """
    Use Bedrock's model to generate a vector-like response.
    Modify the prompt in a way that the model returns an embedding or semantic representation.
    :param text: The input text to convert into embeddings
    :param model_id: The Bedrock model ID to use for embedding generation
    :return: Vector-like response (mock embeddings from Bedrock)
    """
    # Craft the prompt to get a vector-like output (this is not direct embedding)
    # The prompt can be modified as needed to get a more vector-like response
    prompt = f"Convert the following text into a numerical vector representation:\n\nText: {text}"
    
    # Call the Bedrock model
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"input": prompt})
    )
    
    # Process the response
    result = json.loads(response['body'].read())
    return result['completions'][0]['data']





# Function to retrieve documents from OpenSearch using vector search
def retrieve_documents(query_embedding, index_name='your-index', top_k=3):
    """
    Retrieve top_k documents from OpenSearch using vector search.
    :param query_embedding: The query embedding for vector search
    :param index_name: The OpenSearch index to search
    :param top_k: Number of top documents to retrieve
    :return: List of retrieved documents
    """
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding_field_name": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        }
    }

    response = opensearch_client.search(index=index_name, body=search_body)
    documents = [hit["_source"] for hit in response["hits"]["hits"]]
    return documents

# Function to generate text using AWS Bedrock with augmented prompt
def generate_text_with_bedrock(augmented_prompt, model_id="anthropic.claude-v1"):
    """
    Use Bedrock's LLM to generate a response from the augmented prompt.
    :param augmented_prompt: The prompt augmented with retrieved documents
    :param model_id: The ID of the Bedrock model
    :return: The generated text
    """
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"input": augmented_prompt})
    )
    
    result = json.loads(response['body'].read())
    return result['completions'][0]['data']

# Main RAG pipeline
def rag_pipeline(query, query_embedding, index_name='your-index', top_k=3):
    # Step 1: Retrieve relevant documents using vector search
    documents = retrieve_documents(query_embedding, index_name, top_k)

    # Step 2: Augment the query with retrieved documents
    augmented_prompt = f"Query: {query}\n\nRelevant Documents:\n"
    for doc in documents:
        augmented_prompt += f"- {doc['text']}\n"
    
    # Step 3: Use Bedrock to generate a response
    generated_response = generate_text_with_bedrock(augmented_prompt)
    
    return generated_response

# Example usage
query = "What are the benefits of cloud computing?"
query_embedding = generate_embeddings_bedrock(query)

# This should be your query embedding obtained from an embedding model
# query_embedding = [0.1, 0.2, 0.3, ...]  # Replace with the actual embedding

response = rag_pipeline(query, query_embedding, index_name='your-index', top_k=3)
print("Generated Response:", response)
