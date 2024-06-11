import os
import subprocess
import sys

def install_packages():
    required_packages = [
        "llama-index", "llama-index-core", "llama-index-embeddings-openai", "llama-parse",
        "llama-index-vector-stores-kdbai", "pandas", "llama-index-postprocessor-cohere-rerank",
        "kdbai_client", "llama_index.vector_stores.qdrant", "qdrant_client", "llama_index.embeddings.fastembed",
        "llama_index.llms.groq", "chainlit"
    ]
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import asyncio
    import nest_asyncio
    import pandas as pd
    import qdrant_client
    from llama_index.core.memory import ChatMemoryBuffer
    from getpass import getpass
    from llama_parse import LlamaParse
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.llms.groq import Groq
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    import chainlit as cl
    from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.service_context import ServiceContext
except ModuleNotFoundError:
    install_packages()
    import nest_asyncio
    import pandas as pd
    import qdrant_client
    from llama_index.core.memory import ChatMemoryBuffer
    from getpass import getpass
    from llama_parse import LlamaParse
    from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.llms.groq import Groq
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    import chainlit as cl
    from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.service_context import ServiceContext

# Apply nested asyncio to avoid event loop issues
nest_asyncio.apply()

# Function to set up the document parser
def setup_parser(api_key, file_path, parsing_instructions):
    parser = LlamaParse(api_key=api_key, result_type="markdown", parsing_instructions=parsing_instructions)
    documents = parser.load_data(file_path)
    return documents

# Function to set up the vector store
def setup_vector_store(qdrant_url, qdrant_api_key):
    client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name='qdrant_rag')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

# Function to set up the vector store
def setup_vector_store2(qdrant_url, qdrant_api_key):
    client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name='qdrant_rag_upload')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

# Function to set up the embedding model and LLM
def setup_models(embed_model_name, llm_model_name, llm_api_key):
    embed_model = FastEmbedEmbedding(model_name=embed_model_name)
    llm = Groq(model=llm_model_name, api_key=llm_api_key)
    Settings.embed_model = embed_model
    Settings.llm = llm

# Function to create the index
def create_index(documents, storage_context):
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model)
    nodes = splitter(documents)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
    return index

# Function to query the index
#def query_index(index, query):
#    query_engine = index.as_query_engine()
#    response = query_engine.query(query)
#    return response
#



llamaparse_api_key = "llx-IQe4f7HZvGtTkfwNtB6eE2gWvJRH1y2a8esXc6xSjbef170c"
file_path = "RAG_news_DB.pdf"
parsing_instructions = ''''''

documents = setup_parser(llamaparse_api_key, file_path, parsing_instructions)

qdrant_url = "https://23923ac3-c72c-425d-b6fa-2e901980c119.us-east4-0.gcp.cloud.qdrant.io"
qdrant_api_key = "5DwuUxjsKn44qVlFX8qCN20sYdlpQ1ndrebYLJdx-IOeNxo7yHUnIA"
storage_context = setup_vector_store(qdrant_url, qdrant_api_key)

embed_model_name = "BAAI/bge-base-en-v1.5"
llm_model_name = "Llama3-70b-8192"
llm_api_key = "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"
setup_models(embed_model_name, llm_model_name, llm_api_key)

index = create_index(documents, storage_context)

memory = ChatMemoryBuffer.from_defaults(token_limit=150000)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, and you will give answers within 2-3 lines maximum. Take context from previous query and if you don't know the answer just say you don't know"
        "If you are asked for Financial or Investment advise just add at the end that you are not a Financial Expert and Fidelity is not responsible for any loss."
    ),
)


async def query_index_async(index, query):


    response = chat_engine.stream_chat(query)

    return response


# Min execution
@cl.on_chat_start
async def start():
    #query_engine = index.as_query_engine()
    #cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content= "Hello! Im an AI assistant. How may I help you?"
    ).send()



    



@cl.on_message
async def main(message: cl.Message):
    global index
    query = message.content

    file = message.elements
    if file:
        pdf_files = [file for file in message.elements if "pdf" in file.mime]
        pdf_path = (pdf_files[0].path)
        
        documents_file = setup_parser(llamaparse_api_key, pdf_path, parsing_instructions)
        storage_context= setup_vector_store(qdrant_url, qdrant_api_key)
        setup_models(embed_model_name, llm_model_name, llm_api_key)
        index = create_index(documents_file, storage_context)
        if query:
            response = await query_index_async(index, query)
        
            response_message = cl.Message(content="")
            for token in response.response_gen:
                await response_message.stream_token(token=token)

        
        # Send the response back to the user
            await response_message.send()
        

    else:
        # Run the query asynchronously
        response = await query_index_async(index, query)
        
        response_message = cl.Message(content="")
        for token in response.response_gen:
            await response_message.stream_token(token=token)

        
        # Send the response back to the user
        await response_message.send()    

    #query_engine = cl.user_session.get("query_engine")
    #msg = cl.Message(content="", author="Assistant")
    #
    #res = await cl.make_async(query_engine.query)(message.content)
    #print(res)
    #for token in res:
    #    await msg.stream_token(token)
    #await msg.send()    






if __name__ == "__main__":
    # Start the Chainlit server
    main()
