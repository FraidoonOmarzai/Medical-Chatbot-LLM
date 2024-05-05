from src.utils import Utils
from langchain.vectorstores import Pinecone as PineconeLang
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
utils = Utils()


# SECRETS
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
CLOUD = os.environ.get('CLOUD')
REGION = os.environ.get('REGION')


# CREATE INDEX
pc = PineconeClient(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=CLOUD,
                      region=REGION)

index_name = 'mchatbot'

# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=384,  # dimensionality of text-embedding-ada-002
        metric='cosine',
        spec=spec
    )
# connect to index
index = pc.Index(index_name)
# view index stats
index.describe_index_stats()


# Load pdf, text chunks and embedding
data = utils.load_pdf('data/')
text_chunks = utils.get_text_chunks(data)
embeddings = utils.download_hugging_face_embeddings()


# Creating Embeddings for Each of The Text Chunks & store it
vectorstore = PineconeLang.from_texts([t.page_content for t in text_chunks],
                                      embeddings,
                                      index_name=index_name)
