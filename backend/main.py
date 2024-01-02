from fastapi import FastAPI, Query
from typing_extensions import Annotated
from typing import Union, Any
from pydantic import BaseModel, Field, BaseSettings
from fastapi.responses import JSONResponse
from backend.matching_engine_bigtable import MatchingEngine
from backend.embedding_palm import VertexEmbeddings
from langchain.document_loaders import GCSFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from knowledge_bot import change_model
from chat_bot import chat_bot
import os
import uvicorn

class Settings(BaseSettings):
    project_id: str = "project-kangwe-poc"
    project_number: str = "725014442001"
    region: str = "us-central1"
    embedding_dir: str = "llm_embedding_demo"
    me_index_id: str = "3787905518618542080"
    me_index_endpoint_id: str = "7429628767301009408"
    bt_instance_id: str = "langchain"
    request_per_minute: int = 300
    es_datastore_id: str = "langchain-vector-store-faq"

settings = Settings()

os.environ["ME_PROJECT_ID"] = settings.project_id
os.environ["ME_LOCATION"] = settings.region
os.environ["ES_PROJECT_ID"] = settings.project_id
os.environ["ES_LOCATION"] = settings.region
os.environ["DATA_STORE_ID"] = settings.es_datastore_id

# initialize Vetex TextEmbeddings
embedding = VertexEmbeddings(requests_per_minute=settings.request_per_minute)

# initialize vector store
me = MatchingEngine.from_components(
    project_id=settings.project_id,
    region=settings.region,
    gcs_bucket_name=settings.embedding_dir,
    index_id=f"projects/{settings.project_id}/locations/{settings.region}/indexes/{settings.me_index_id}",
    endpoint_id=f"projects/{settings.project_id}/locations/{settings.region}/indexEndpoints/{settings.me_index_endpoint_id}",
    instance_id=settings.bt_instance_id,
    embedding=embedding
)

# initialize knowledage bot
kb = change_model(model_name="palm", prompt_type="native" ,vectordb=me)

# initialize chat bot
cb = chat_bot(model_name="text_palm", matching_engine_instance=me, data_folder='./content')

roles = []

app = FastAPI()

class RootResponse(BaseModel):
    ok: bool

class SearchEmbeddingRequest(BaseModel):
    doc: str = Field(description="The doc text is used to do embedding search", min_length=1, max_length=1000)
    k: int = Field(default=4 ,description="The amount of neighbors that will be retrieved", ge=1, le=10)
    collection: Union[str, None] = Field(default=None, description="The VectorDB collection filter")

class SearchEmbeddingResponse(BaseModel):
    docs: list[str] = Field(default=[], description="The matching docs")

class ListCollectionResponse(BaseModel):
    collections: list[str] = Field(default=[], description="The collection names")

class DocRequest(BaseModel):
    doc: str = Field(description="The doc text is used to add into VectorDB", min_length=1, max_length=1000)
    collection: Union[str, None] = Field(default=None, description="The VectorDB collection filter")

class DocResponse(BaseModel):
    success: bool = Field(description="result of the doc add request")
    description: str = Field(default="" ,description="detail description of the response")

class DocBatchRequest(BaseModel):
    doc_location: str = Field(description="The Google Cloud Storage object path")
    collection: Union[str, None] = Field(default=None, description="The VectorDB collection filter")

class DocBatchReponse(BaseModel):
    success: bool = Field(description="result of the doc delete request")
    description: str = Field(default="" ,description="detail description of the response")

class CollectionRequest(BaseModel):
    collection: Union[str, None] = Field(default=None, description="The VectorDB collection filter")

class CollectionReponse(BaseModel):
    success: bool = Field(description="result of the collection delete request")
    description: str = Field(default="" ,description="detail description of the response")

class Message(BaseModel):
    message: str

class ChangeModelRequest(BaseModel):
    model_name: str = Field(description="The LLM Model name")
    prompt_type: Union[str, None] = Field(default=None, description="The prompt type")
    collection: Union[str, None] = Field(default=None, description="The VectorDB collection filter")

class ChangeModelResponse(BaseModel):
    success: bool = Field(description="result of the change model request")
    description: str = Field(default="" ,description="detail description of the response")

class ListRoleResponse(BaseModel):
    roles: list[str] = Field(default=[], description="The role names")

class ChatbotSendMessageRequest(BaseModel):
    role: str = Field(description="The role the chatbot act")
    model: Union[str, None] = Field(default=None, description="The LLM Model name")
    message: str = Field(description="The message sent to chatbot")

class ChatbotSendMessageReponse(BaseModel):
    response_message: str = Field(description="The message chatbot send to client")
    debug_message: str = Field(description="The chatbot debug message")

class ChatbotResetMessageRequest(BaseModel):
    role: str = Field(description="The role the chatbot act")

class ChatbotResetMessageResponse(BaseModel):
    success: bool = Field(description="result of the chatbot reset message request")

class KBbotSendMessageRequest(BaseModel):
    message: str = Field(description="The message sent to kbbot")

class KBbotSendMessageReponse(BaseModel):
    response_message: str = Field(description="The message kbbot send to client")
    debug_message: str = Field(description="The chatbot debug message")

@app.get("/")
async def root() -> RootResponse:
    return RootResponse(ok=True)

@app.post("/search", response_model=SearchEmbeddingResponse, responses={404: {"model": Message}})
async def search_embedding(
    req: SearchEmbeddingRequest, 
    vectordbtype: Annotated[str, Query(description="the type of vectorDB")] = "matchengine"
) -> Any:
    docs = []
    doc = req.doc
    top_k = req.k

    if req.collection != None:
        collection_name = req.collection
    else:
        collection_name = ""

    try:
        result = me.similarity_search(query=doc, k=top_k, collection=collection_name)
    except Exception as error:
        return JSONResponse(status_code=404, content={"message": error})
    
    for result_doc in result:
        docs.append(result_doc.page_content)

    return SearchEmbeddingResponse(docs=docs)

@app.post("/doc")
async def doc_adding(
    req: DocRequest, 
    vectordbtype: Annotated[str, Query(description="the type of vectorDB")] = "matchengine"
) -> Any:
    success = False
    doc = req.doc

    if req.collection != None:
        collection_name = req.collection
    else:
        collection_name = ""

    try:
        me.add_texts(texts=[doc], collection=collection_name)
    except Exception as error:
        return DocResponse(success=success, description=error)
    
    success = True
    return DocResponse(success=success)

@app.delete("/doc")
async def doc_deleting(
    req: DocRequest,
    vectordbtype: Annotated[str, Query(description="the type of vectorDB")] = "matchengine"
) -> Any:
    success = False
    doc = req.doc

    if req.collection != None:
        collection_name = req.collection
    else:
        collection_name = ""

    try:
        me.delete_texts(texts=[doc], collection=collection_name)
    except Exception as error:
        return DocResponse(success=success, description=error)
    
    success = True
    return DocResponse(success=success)

@app.post("/docsbatch")
async def docs_batch_adding(
    req: DocBatchRequest,
    vectordbtype: Annotated[str, Query(description="the type of vectorDB")] = "matchengine"
) -> Any:
    success = False
    documents = []

    if req.collection != None:
        collection_name = req.collection
    else:
        collection_name = ""

    bucket_name = req.doc_location.split("/")[2]
    blob_name = "/".join(req.doc_location.split("/")[3:])

    loader = GCSFileLoader(project_name=settings.project_id, bucket=bucket_name, blob=blob_name)
    documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(documents)

    print("split")

    texts = [doc.page_content for doc in doc_splits]
    try:
        me.add_texts(texts=texts, collection=collection_name)
    except Exception as error:
        return DocBatchReponse(success=success, description=error)
    
    success = True
    return DocBatchReponse(success=success)

@app.delete("/docsbatch")
async def docs_batch_deleting(
    req: DocBatchRequest,
    vectordbtype: Annotated[str, Query(description="the type of vectorDB")] = "matchengine"
) -> Any:
    success = False
    documents = []

    if req.collection != None:
        collection_name = req.collection
    else:
        collection_name = ""

    bucket_name = req.doc_location.split("/")[2]
    blob_name = "/".join(req.doc_location.split("/")[3:])

    loader = GCSFileLoader(project_name=settings.project_id, bucket=bucket_name, blob=blob_name)
    documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(documents)
    print("split")
    texts = [doc.page_content for doc in doc_splits]
    try:
        me.delete_texts(texts=texts, collection=collection_name)
    except Exception as error:
        return DocBatchReponse(success=success, description=error)
    
    success = True
    return DocBatchReponse(success=success)

@app.get("/collection")
async def collection_listing(
    vectordbtype: Annotated[str, Query(description="the type of vectorDB")] = "matchengine"
) -> Any:
    collections = me.list_collection()
    return ListCollectionResponse(collections=collections)

@app.delete("/collection")
async def collection_delete(
    req: CollectionRequest
) -> Any:
    success = False

    if req.collection != None:
        collection_name = req.collection
    else:
        collection_name = ""

    try:
        me.delete_collection(collection=collection_name)
    except Exception as error:
        return CollectionReponse(success=success, description=error)
    
    success = True
    return CollectionReponse(success=success)
    
@app.post("/chatbot/changemodel")
async def chatbot_changemodel(
    req: ChangeModelRequest
) -> Any:
    success = False
    # add the chatbot class call funtion here.
    try:
        cb.change_model(req.model_name)
    except Exception as error:
        return ChangeModelResponse(success=success, description=error)
    
    success = True
    return ChangeModelResponse(success=success)

# @app.get("/chatbot/changememorytype")

@app.get("/chatbot/list/role")
async def chatbot_listrole() -> Any:
    global roles
    if roles == []:
        roles = cb.list_role()
    else:
        pass
    return ListRoleResponse(roles=roles)

@app.post("/chatbot/sendmessage")
async def chatbot_sendmessage(
    req: ChatbotSendMessageRequest
) -> Any:
    response_message = ""
    debug_message = ""
    # add the chatbot class call funtion here.
    try:
        response_message, debug_message = cb.send_message(req.role, req.message)
    except Exception as error:
        debug_message = str(error)
    return ChatbotSendMessageReponse(response_message=response_message, debug_message=debug_message)

@app.post("/chatbot/resetmessage")
async def chatbot_resetmessage(
    req: ChatbotResetMessageRequest
) -> Any:
    success = True
    # add the chatbot class call funtion here.
    try:
        cb.reset_history(req.role)
    except Exception as error:
        success = False
    return ChatbotResetMessageResponse(success=success)

@app.post("/kbbot/changemodel")
async def kbbot_changemodel(
    req: ChangeModelRequest
) -> Any:
    global kb
    success = False
    model_name = req.model_name
    # add the kbbot class call funtion here.
    if model_name == "palm":        
        if req.collection != None:
            collection_name = req.collection
        else:
            collection_name = ""
        
        if req.prompt_type != None:
            prompt_type = req.prompt_type
        else:
            prompt_type = "native"
        
        try: 
            kb = change_model(model_name=model_name, prompt_type=prompt_type, vectordb=me, collection=collection_name)
        except Exception as error:
            return ChangeModelResponse(success=success, description=error)
    else:
        try: 
            kb = change_model(model_name="enterprise_search")
        except Exception as error:
            return ChangeModelResponse(success=success, description=error)

    success = True
    return ChangeModelResponse(success=success)

@app.post("/kbbot/sendmessage")
async def kbbot_sendmessage(
    req: KBbotSendMessageRequest
) -> Any:
    response_message = ""
    debug_message = ""
    # add the kbbot class call funtion here.
    try:
        response_message, debug_message = kb.send_message(req.message)
    except Exception as error:
        debug_message = str(error)
    return KBbotSendMessageReponse(response_message=response_message, debug_message=debug_message)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)