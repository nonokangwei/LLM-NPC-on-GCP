"""Vertex Matching Engine implementation of the vector store."""
from __future__ import annotations

import json
import logging
import time
import uuid, hashlib
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Type

from langchain.docstore.document import Document
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from google.cloud import storage
from google.cloud import bigtable
import google.cloud.bigtable.row_filters as row_filters
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1beta1 import FindNeighborsResponse
from google.oauth2.service_account import Credentials
from google.cloud import aiplatform_v1

logger = logging.getLogger()
bigtable_table_id = "vectordb"
bigtable_column_family_id = "vector_text"
collection_list = ["default"]
ENDPOINT = "{}-aiplatform.googleapis.com".format("us-central1")

class MatchingEngine(VectorStore):
    """Vertex Matching Engine implementation of the vector store.

    While the embeddings are stored in the Matching Engine, the embedded
    documents will be stored in GCS.

    An existing Index and corresponding Endpoint are preconditions for
    using this module.

    See usage in docs/modules/indexes/vectorstores/examples/matchingengine.ipynb

    Note that this implementation is mostly meant for reading if you are
    planning to do a real time implementation. While reading is a real time
    operation, updating the index takes close to one hour."""

    def __init__(
        self,
        project_id: str,
        index: MatchingEngineIndex,
        endpoint: MatchingEngineIndexEndpoint,
        embedding: Embeddings,
        gcs_client: storage.Client,
        gcs_bucket_name: str,
        index_client: aiplatform_v1.IndexServiceClient,
        index_endpoint_client: aiplatform_v1.IndexEndpointServiceClient,
        bigtable_client: bigtable.Client,
        bigtable_instance_id: str,
        credentials: Optional[Credentials] = None,
    ):
        """Vertex Matching Engine implementation of the vector store.

        While the embeddings are stored in the Matching Engine, the embedded
        documents will be stored in GCS.

        An existing Index and corresponding Endpoint are preconditions for
        using this module.

        See usage in
        docs/modules/indexes/vectorstores/examples/matchingengine.ipynb.

        Note that this implementation is mostly meant for reading if you are
        planning to do a real time implementation. While reading is a real time
        operation, updating the index takes close to one hour.

        Attributes:
            project_id: The GCS project id.
            index: The created index class. See
                ~:func:`MatchingEngine.from_components`.
            endpoint: The created endpoint class. See
                ~:func:`MatchingEngine.from_components`.
            embedding: A :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
            gcs_client: The GCS client.
            gcs_bucket_name: The GCS bucket name.
            credentials (Optional): Created GCP credentials.
        """
        super().__init__()
        self._validate_google_libraries_installation()

        self.project_id = project_id
        self.index = index
        self.endpoint = endpoint
        self.embedding = embedding
        self.bigtable_client = bigtable_client
        self.credentials = credentials
        self.bigtable_instance_id = bigtable_instance_id
        self.gcs_client = gcs_client
        self.gcs_bucket_name = gcs_bucket_name
        self.index_client = index_client
        self.index_endpoint_client = index_endpoint_client

    def _validate_google_libraries_installation(self) -> None:
        """Validates that Google libraries that are needed are installed."""
        try:
            from google.cloud import aiplatform, bigtable, storage  # noqa: F401
            from google.oauth2 import service_account  # noqa: F401
        except ImportError:
            raise ImportError(
                "You must run `pip install --upgrade "
                "google-cloud-aiplatform google-cloud-storage google-cloud-bigtable`"
                "to use the MatchingEngine Vectorstore."
            )
    
    def list_collection(
        self,
        **kwargs: Any,
    )-> List[str]:
        return collection_list

    def add_texts(
        self,
        texts: Iterable[str],
        collection: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        logger.debug("Embedding documents.")
        embeddings = self.embedding.embed_documents(list(texts))
        jsons = []
        ids = []
        global collection_list

        if collection in collection_list:
            pass
        else:
            collection_list.append(collection)
        
        # Could be improved with async.
        for embedding, text in zip(embeddings, texts):
            # id = str(uuid.uuid4())
            id = hashlib.sha256((str(text)+str(collection)).encode('utf-8')).hexdigest()
            ids.append(id)
            jsons.append({"id": id, "embedding": embedding, "text": text})
            # self._upload_to_gcs(text, f"documents/{id}")

        # logger.debug(f"Uploaded {len(ids)} documents to GCS.")

        # # Creating json lines from the embedded documents.
        # result_str = "\n".join([json.dumps(x) for x in jsons])

        # filename_prefix = f"indexes/{uuid.uuid4()}"
        # filename = f"{filename_prefix}/{time.time()}.json"
        # self._upload_to_gcs(result_str, filename)
        # logger.debug(
        #     f"Uploaded updated json with embeddings to "
        #     f"{self.gcs_bucket_name}/{filename}."
        # )

        # self.index = self.index.update_embeddings(
        #     contents_delta_uri=f"gs://{self.gcs_bucket_name}/{filename_prefix}/"
        # )
        self._stream_upsert_vme(data=jsons, collection=collection)
        logger.debug("Updated index with new configuration.")

        return ids
    
    def delete_texts(
        self,
        texts: Iterable[str],
        collection: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and delete from the vectorstore.

        Args:
            texts: Iterable of strings to delete from the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from deleting the texts into the vectorstore.
        """
        logger.debug("Delete embedding documents.")
        ids = []
        jsons =[]
        # Could be improved with async.

        for text in texts:
            # id = str(uuid.uuid4())
            id = hashlib.sha256((str(text)+str(collection)).encode('utf-8')).hexdigest()
            ids.append(id)
            jsons.append({"id": id})

        self._stream_delete_vme(data=jsons, collection=collection)
        logger.debug("Delete index with new configuration.")

        return ids
    
    def delete_collection(
        self,
        collection: str,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        logger.debug("Delete embedding documents by collection")
        ids = []
        jsons = []

        instance_client = self.bigtable_client.instance(self.bigtable_instance_id)
        table_client = instance_client.table(bigtable_table_id)
        collection_name = collection

        try:
            rows = table_client.read_rows(
                filter_=row_filters.ValueRegexFilter(collection_name.encode("utf-8"))
            )
        except Exception as error:
            logger.debug(f"Fail to fetech collection data {collection_name} with error {error}")

        for row in rows:
            id = row.row_key.decode("utf-8")
            ids.append(id)
            jsons.append({"id": id})
        
        if jsons.__len__() > 0:
            self._stream_delete_vme(data=jsons, collection=collection_name)
            logger.debug("Delete index with new configuration.")

        return ids
    
    def _read_text(self, row_key_text: str) -> str:

        instance_client = self.bigtable_client.instance(self.bigtable_instance_id)
        table_client = instance_client.table(bigtable_table_id)

        row_key = row_key_text
        row = table_client.read_row(row_key)
        column_family_id = bigtable_column_family_id
        buf = row.cell_value(column_family_id=column_family_id, column="text".encode('utf-8'))
        
        if buf != None:
            return buf.decode("utf-8")
        else:
            return None

    def _upload_to_gcs(self, data: str, gcs_location: str) -> None:
        """Uploads data to gcs_location.

        Args:
            data: The data that will be stored.
            gcs_location: The location where the data will be stored.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        blob = bucket.blob(gcs_location)
        blob.upload_from_string(data)
    
    def _stream_upsert_vme(self, data: List[dict], collection: str) -> None:
        """Stream upsert data to vertex matching engine
        
        Args:
            data: The data will be stored [{"id": id, "embedding": embedding}]
        """
        from google.cloud import aiplatform_v1
        import datetime

        instance_client = self.bigtable_client.instance(self.bigtable_instance_id)
        table_client = instance_client.table(bigtable_table_id)
        column_family_id = bigtable_column_family_id
        
        # collection_name = collection
        
        if collection != "":
            restriction  = aiplatform_v1.IndexDatapoint.Restriction(
                {
                    "namespace" : "collection",
                    "allow_list" : [collection]
                }
            )
            collection_name = collection
        else:
            restriction  = aiplatform_v1.IndexDatapoint.Restriction({})
            collection_name = "default"

        for embedding_data in data:
            datapoint = aiplatform_v1.IndexDatapoint(
                datapoint_id=embedding_data["id"],
                feature_vector=embedding_data["embedding"],
                restricts=[restriction]
            )

            upsert_datapoint_request = aiplatform_v1.UpsertDatapointsRequest(
                        index=self.index.name,
                        datapoints=[datapoint]
                    )

            response = self.index_client.upsert_datapoints(request=upsert_datapoint_request)
            if response != {}:
                logger.debug(f"Fail to upsert {embedding_data['id']} embedding data.")
            
            try:
                timestamp = datetime.datetime.utcnow()
                row_key = embedding_data["id"]
                row = table_client.direct_row(row_key)
                row.set_cell(column_family_id, "text".encode("utf-8"), embedding_data["text"].encode("utf-8"), timestamp)
                row.set_cell(column_family_id,"collection".encode("utf-8"), collection_name, timestamp)
                row.commit()
            except Exception as error:
                print(error)
                logger.debug(f"Fail to upsert {embedding_data['id']} with error {error}")

    def _stream_delete_vme(self, data: List[str], collection: str) -> None:
        """Stream upsert data to vertex matching engine
        
        Args:
            data: The data will be stored [{"id": id}]
        """

        instance_client = self.bigtable_client.instance(self.bigtable_instance_id)
        table_client = instance_client.table(bigtable_table_id)
        
        for id in data:
            delete_datapoint_request = aiplatform_v1.RemoveDatapointsRequest(
               index=self.index.name,
               datapoint_ids=[id['id']]
            )

            response = self.index_client.remove_datapoints(request=delete_datapoint_request)
            if response != None:
                logger.debug(f"Fail to delete {id['id']} embedding data.")
            
            try:
                row_key = id['id']
                row = table_client.row(row_key)
                row.delete()
                row.commit()
            except Exception as error:
                print(error)
                logger.debug(f"Fail to delete {id['id']} with error {error}")
        
        global collection_list

        if collection != "":
            collection_name = collection
        else:
            collection_name = "default"
        
        rows = table_client.read_rows(
            filter_=row_filters.RowFilterChain(
                filters=[
                    row_filters.ColumnQualifierRegexFilter("collection".encode("utf-8")),
                    row_filters.ValueRegexFilter(collection_name.encode("utf-8")),
                    row_filters.CellsColumnLimitFilter(1),
                    row_filters.StripValueTransformerFilter(True),
                ]
            ) 
        )
        rows.consume_all()
        if len(rows.rows) == 0:
            collection_list.remove(collection_name)
        else:
            pass

    def get_matches(
            self,
            embeddings: List[str],
            n_matches: int,
            collection: str) -> FindNeighborsResponse:
        '''
        get matches from matching engine given a vector query
        Uses public endpoint

        '''

        client_options = {
              "api_endpoint": self.endpoint.public_endpoint_domain_name
        }

        vertex_ai_client = aiplatform_v1beta1.MatchServiceClient(
              client_options=client_options,
        )

        request = aiplatform_v1beta1.FindNeighborsRequest(
              index_endpoint=self.endpoint.resource_name,
              deployed_index_id=self.endpoint.deployed_indexes[0].id,
          )
        
        if collection != "":
            restriction  = aiplatform_v1beta1.IndexDatapoint.Restriction(
                {
                    "namespace" : "collection",
                    "allow_list" : [collection]
                }
            )
        else:
            restriction  =  aiplatform_v1beta1.IndexDatapoint.Restriction(
                {}
            )
        
        for i, embedding in enumerate(embeddings):
            query = aiplatform_v1beta1.FindNeighborsRequest.Query(
                datapoint=aiplatform_v1beta1.IndexDatapoint(
                        datapoint_id=str(i),
                        feature_vector=embedding,
                        restricts=[restriction],
                    ),
                neighbor_count = n_matches
            )
            request.queries.append(query)
        
        response = vertex_ai_client.find_neighbors(request)
        return response

    def similarity_search(
        self, query: str, k: int = 4, collection: str = "", **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.

        Returns:
            A list of k matching documents.
        """

        logger.debug(f"Embedding query {query}.")
        embedding_query = self.embedding.embed_documents([query])

        # TO-DO: Pending query sdk integration
        # response = self.endpoint.match(
        #     deployed_index_id=self._get_index_id(),
        #     queries=embedding_query,
        #     num_neighbors=k,
        # )

        response = self.get_matches(embedding_query,
                                    k, collection)
        
        if response != None:
            response = response.nearest_neighbors[0].neighbors
        else:
            raise Exception(f"Failed to query index {str(response)}")

        if len(response) == 0:
            return []

        logger.debug(f"Found {len(response)} matches for the query {query}.")

        results = []

        # I'm only getting the first one because queries receives an array
        # and the similarity_search method only recevies one query. This
        # means that the match method will always return an array with only
        # one element.
        # for doc in response[0]:
        #     page_content = self._download_from_gcs(f"documents/{doc.id}")
        #     results.append(Document(page_content=page_content))
        for doc in response:
            page_content = self._read_text(doc.datapoint.datapoint_id)
            results.append(Document(page_content=page_content))


        logger.debug("Downloaded documents for query.")

        return results

    def _get_index_id(self) -> str:
        """Gets the correct index id for the endpoint.

        Returns:
            The index id if found (which should be found) or throws
            ValueError otherwise.
        """
        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.resource_name:
                return index.id

        raise ValueError(
            f"No index with id {self.index.resource_name} "
            f"deployed on endpoint "
            f"{self.endpoint.display_name}."
        )

    def _download_from_gcs(self, gcs_location: str) -> str:
        """Downloads from GCS in text format.

        Args:
            gcs_location: The location where the file is located.

        Returns:
            The string contents of the file.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        blob = bucket.blob(gcs_location)
        return blob.download_as_string()

    @classmethod
    def from_texts(
        cls: Type["MatchingEngine"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MatchingEngine":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `MatchingEngine.from_components(...)` and then call "
            "`add_texts`"
        )

    @classmethod
    def from_components(
        cls: Type["MatchingEngine"],
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        index_id: str,
        endpoint_id: str,
        instance_id: str,
        credentials_path: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
    ) -> "MatchingEngine":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: The location where the vectors will be stored in
            order for the index to be created.
            index_id: The id of the created index.
            endpoint_id: The id of the created endpoint.
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.

        Returns:
            A configured MatchingEngine with the texts added to the index.
        """
        gcs_bucket_name = cls._validate_gcs_bucket(gcs_bucket_name)
        credentials = cls._create_credentials_from_file(credentials_path)
        index = cls._create_index_by_id(index_id, project_id, region, credentials)
        endpoint = cls._create_endpoint_by_id(
            endpoint_id, project_id, region, credentials
        )

        gcs_client = cls._get_gcs_client(credentials, project_id)
        bigtable_client = cls._get_bigtable_client(credentials, project_id)
        index_client = cls._get_index_client(project_id, region, credentials)
        index_endpoint_client = cls._get_index_endpoint_client(project_id, region, credentials)
        cls._init_aiplatform(project_id, region, gcs_bucket_name, credentials)

        return cls(
            project_id=project_id,
            index=index,
            endpoint=endpoint,
            embedding=embedding or cls._get_default_embeddings(),
            gcs_client=gcs_client,
            index_client=index_client,
            index_endpoint_client=index_endpoint_client,
            credentials=credentials,
            gcs_bucket_name=gcs_bucket_name,
            bigtable_client=bigtable_client,
            bigtable_instance_id=instance_id
        )

    @classmethod
    def _validate_gcs_bucket(cls, gcs_bucket_name: str) -> str:
        """Validates the gcs_bucket_name as a bucket name.

        Args:
              gcs_bucket_name: The received bucket uri.

        Returns:
              A valid gcs_bucket_name or throws ValueError if full path is
              provided.
        """
        gcs_bucket_name = gcs_bucket_name.replace("gs://", "")
        if "/" in gcs_bucket_name:
            raise ValueError(
                f"The argument gcs_bucket_name should only be "
                f"the bucket name. Received {gcs_bucket_name}"
            )
        return gcs_bucket_name

    @classmethod
    def _create_credentials_from_file(
        cls, json_credentials_path: Optional[str]
    ) -> Optional[Credentials]:
        """Creates credentials for GCP.

        Args:
             json_credentials_path: The path on the file system where the
             credentials are stored.

         Returns:
             An optional of Credentials or None, in which case the default
             will be used.
        """

        from google.oauth2 import service_account

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(
                json_credentials_path
            )

        return credentials

    @classmethod
    def _create_index_by_id(
        cls, index_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndex:
        """Creates a MatchingEngineIndex object by id.

        Args:
            index_id: The created index id.
            project_id: The project to retrieve index from.
            region: Location to retrieve index from.
            credentials: GCS credentials.

        Returns:
            A configured MatchingEngineIndex.
        """

        from google.cloud import aiplatform_v1


        logger.debug(f"Creating matching engine index with id {index_id}.")
        index_client = cls._get_index_client(project_id, region, credentials)
        request = aiplatform_v1.GetIndexRequest(name=index_id)
        return index_client.get_index(request=request)

    @classmethod
    def _create_endpoint_by_id(
        cls, endpoint_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndexEndpoint:
        """Creates a MatchingEngineIndexEndpoint object by id.

        Args:
            endpoint_id: The created endpoint id.
            project_id: The project to retrieve index from.
            region: Location to retrieve index from.
            credentials: GCS credentials.

        Returns:
            A configured MatchingEngineIndexEndpoint.
        """

        from google.cloud import aiplatform

        logger.debug(f"Creating endpoint with id {endpoint_id}.")
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=project_id,
            location=region,
            credentials=credentials,
        )

    @classmethod
    def _get_gcs_client(
        cls, credentials: "Credentials", project_id: str
    ) -> "storage.Client":
        """Lazily creates a GCS client.

        Returns:
            A configured GCS client.
        """

        from google.cloud import storage

        return storage.Client(credentials=credentials, project=project_id)
    
    @classmethod
    def _get_bigtable_client(
        cls, credentials: "Credentials", project_id: str
    ) -> "bigtable.Client":
        """Lazily creates a BigTable client.

        Returns:
            A configured BigTable client.
        """

        from google.cloud import bigtable

        return bigtable.Client(credentials=credentials, project=project_id,  admin=True)
    
    @classmethod
    def _get_index_client(
        cls, project_id: str, region: str, credentials: "Credentials"
    ) -> "storage.Client":
        """Lazily creates a Matching Engine Index client.

        Returns:
            A configured Matching Engine Index client.
        """

        from google.cloud import aiplatform_v1

        PARENT = f"projects/{project_id}/locations/{region}"
        ENDPOINT = f"{region}-aiplatform.googleapis.com"
        return aiplatform_v1.IndexServiceClient(
            client_options=dict(api_endpoint=ENDPOINT),
            credentials=credentials
        )

    @classmethod
    def _get_index_endpoint_client(
        cls, project_id: str, region: str, credentials: "Credentials"
    ) -> "storage.Client":
        """Lazily creates a Matching Engine Index Endpoint client.

        Returns:
            A configured Matching Engine Index Endpoint client.
        """

        from google.cloud import aiplatform_v1

        PARENT = f"projects/{project_id}/locations/{region}"
        ENDPOINT = f"{region}-aiplatform.googleapis.com"
        return aiplatform_v1.IndexEndpointServiceClient(
            client_options=dict(api_endpoint=ENDPOINT),
            credentials=credentials
        )

    @classmethod
    def _init_aiplatform(
        cls,
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        credentials: "Credentials",
    ) -> None:
        """Configures the aiplatform library.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: GCS staging location.
            credentials: The GCS Credentials object.
        """

        from google.cloud import aiplatform

        logger.debug(
            f"Initializing AI Platform for project {project_id} on "
            f"{region} and for {gcs_bucket_name}."
        )
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=gcs_bucket_name,
            credentials=credentials,
        )

    @classmethod
    def _get_default_embeddings(cls) -> TensorflowHubEmbeddings:
        """This function returns the default embedding.

        Returns:
            Default TensorflowHubEmbeddings to use.
        """
        return TensorflowHubEmbeddings()