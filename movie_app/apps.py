from django.apps import AppConfig
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore

class MovieAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "movie_app"
    data = pd.read_json("D:\Web Development\Movie_Synopsis_Search\static_files\datasets\imdb_movies_processed.json")
    predictor = tf.keras.models.load_model("D:\Web Development\Movie_Synopsis_Search\static_files\models\movie_intent_recognition.h5",{'KerasLayer':hub.KerasLayer}) 
    tr = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    document_store = FAISSDocumentStore.load(index_path="D:\Web Development\Movie_Synopsis_Search\movies_doc_store.faiss",
                                             config_path="D:\Web Development\Movie_Synopsis_Search\movies_doc_store.json")
    
    retriever = DensePassageRetriever(document_store=document_store,query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base", use_gpu=True, embed_title=True,)
    
    