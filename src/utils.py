from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# Importing embeddings from langchain-community instead of langchain
from langchain_community.embeddings import HuggingFaceEmbeddings



class Utils:
    """Class utils for loading some methods
    """

    def __init__(self):
        pass

    # load pdf files
    def load_pdf(data):
        loader = DirectoryLoader(data,
                                 glob="*.pdf",
                                 loader_cls=PyPDFLoader)

        documents = loader.load()

        return documents

    # crete chunks
    def text_split(extracted_data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(extracted_data)

        return text_chunks

    # download embedding model
    def download_hugging_face_embeddings():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

        return embeddings
