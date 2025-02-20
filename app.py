from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

app = Flask(__name__)

# Load sensitive keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "test-new"

if not (OPENAI_API_KEY and PINECONE_API_KEY and PINECONE_ENVIRONMENT):
    raise EnvironmentError("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENVIRONMENT as environment variables.")

# Initialize Pinecone and embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
index = pc.Index(INDEX_NAME)

bm25_encoder = BM25Encoder().load("all_bm25_value.json")  # Update path to BM25 file

vectorstore = PineconeHybridSearchRetriever(
    alpha=0.5, embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, top_k=20)

def create_chatbot_retrieval_qa():
    prompt_template = """
Note:
- You are a helpful assistant focused on providing relevant answers.
- Avoid answering questions that are not related to the context.
- Use the provided metadata for location and contact information to enhance your response.
- Analyze the company descriptions and provide the top 3 companies based on the question asked.

Context: {context}

Question: {question}

Provide the best companies that fit the criteria outlined in the question above, based on the given company descriptions.
"""


    after_rag_prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY) # type: ignore

    def hyde_retriever(query):
        return vectorstore.get_relevant_documents(query)

    chain = (
        {"context": hyde_retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain

@app.route("/chat", methods=["POST"])
def chatbot_api():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request payload."}), 400

        query = data.get("query")
        if not query or query.isspace():
            return jsonify({"error": "Query is required."}), 400

        chatbot = create_chatbot_retrieval_qa()
        response = chatbot.invoke(query)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
