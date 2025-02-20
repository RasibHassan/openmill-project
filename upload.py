import traceback
import logging
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import os
import time
import re
from dotenv import load_dotenv
import pymssql
import warnings
warnings.filterwarnings("ignore")
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='embedding_process.log'
)

def update_embedding_status():
    try:
        connection = pymssql.connect(
            host='3.15.185.195',
            user='app_ai_access',
            password='o747yDe&0A-n@OPX59',
            database='OpenMill_AI'
        )
        with connection.cursor() as cursor:
            update_query = "UPDATE company SET embeddingStatus = 1 WHERE embeddingStatus = 0 AND bioLong IS NOT NULL;"
            cursor.execute(update_query)
            connection.commit()
            logging.info("Updated embeddingStatus for processed records.")
    except Exception as e:
        logging.error(f"Error updating embeddingStatus: {e}")
        logging.error(traceback.format_exc())
        raise
    finally:
        connection.close()

def embed_documents_in_pinecone(chunks, metadata, index_name):
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        bm25_encoder = BM25Encoder().load("all_bm25_value.json")
        index = pc.Index(index_name)
        embeder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        vectorstore = PineconeHybridSearchRetriever(
            embeddings=embeder, sparse_encoder=bm25_encoder, index=index, top_k=10
        )
        
        try:
            logging.info(f"Adding {len(chunks)} chunks to Pinecone...")
            vectorstore.add_texts(chunks, metadatas=metadata)
            logging.info("Chunks added successfully.")
        except Exception as e:
            logging.error(f"An error occurred while adding texts to Pinecone: {str(e)}")
            return None  # Return None if there was an error during addition

        # Update embedding status
        update_embedding_status()
        logging.info("Embedding status updated successfully.")
        return vectorstore

    except Exception as e:
        logging.error(f"Pinecone embedding error: {e}")
        logging.error(traceback.format_exc())
        return None

def process_database():
    try:
        connection = pymssql.connect(
            host='3.15.185.195',
            user='app_ai_access',
            password='o747yDe&0A-n@OPX59',
            database='OpenMill_AI'
        )

        with connection.cursor() as cursor:
            query = "SELECT bioLong, phone, email, website, street1, city, state, country FROM company WHERE embeddingStatus = 0 AND bioLong IS NOT NULL;"
            cursor.execute(query)
            results = cursor.fetchall()
            
            if not results:
                logging.info("No new records to process.")
                return [], []

            columns = [desc[0] for desc in cursor.description]
            data = [dict(zip(columns, row)) for row in results]

            bioLong_list = [row["bioLong"] if row["bioLong"] else "N/A" for row in data]
            cleaned_texts = [re.sub(r'<p>|</p>', '', text) for text in bioLong_list]

            contact_location_list = [
                {
                    "contact info": f'phone: {row["phone"]}  email: {row["email"]}  website: {row["website"]}',
                    "Location": f'street: {row["street1"]}  city: {row["city"]}  state: {row["state"]}  country: {row["country"]}'
                }
                for row in data
            ]

            logging.info(f"Processed {len(cleaned_texts)} records.")
            logging.info(f"Sample record: {cleaned_texts[0]}")
            logging.info("Database processing completed.")
            return cleaned_texts, contact_location_list

    except Exception as e:
        logging.error(f"Database processing error: {e}")
        logging.error(traceback.format_exc())
        return [], []

    finally:
        connection.close()




def main():
    while True:
        try:
            logging.info("Starting document processing...")
            cleaned_texts, contact_location_list = process_database()
            
            if cleaned_texts and contact_location_list:
                embed_result = embed_documents_in_pinecone(cleaned_texts, contact_location_list, "test-new")
                
                if embed_result:
                    logging.info("Embedding completed successfully.")
                else:
                    logging.warning("Embedding process encountered issues.")
            
            logging.info("Waiting for next processing cycle...")
            time.sleep(120)  # Wait a minute before retrying

        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            logging.error(traceback.format_exc())
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    load_dotenv()
    main()