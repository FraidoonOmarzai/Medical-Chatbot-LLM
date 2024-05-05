from flask import Flask, render_template, request
from langchain.vectorstores import Pinecone as PineconeLang
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.utils import Utils
from src.constants import *
from src.prompts import *
import os

from langchain_community.embeddings import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore")


load_dotenv()
utils = Utils()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


app = Flask(__name__)


# embeddings = utils.download_hugging_face_embeddings()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Loading the index
vectorstore = PineconeLang.from_existing_index(INDEX_NAME, embeddings)


PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}


llm = CTransformers(model=MODEL_PATH,
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})


qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever(
                                     search_kwargs={'k': 2}),
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
