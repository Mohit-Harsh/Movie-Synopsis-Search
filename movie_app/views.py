from django.shortcuts import render
from django.http import HttpRequest,request
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from haystack.document_stores import FAISSDocumentStore
import sys
from .models import Movie
from .apps import MovieAppConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from haystack.nodes import DensePassageRetriever
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline

def filter_indices1(x):
    if(x[1]>=0.7):
        return True
    else:
        return False
    
def filter_indices2(x):
    if(x[1]>=0.5):
        return True
    else:
        return False

def search_by_synopsis(question):
    
    print("Called search_by_synopsis")
    
    df = MovieAppConfig.data
    
    document_store = MovieAppConfig.document_store
    
    retriever = MovieAppConfig.retriever
    
    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query=str(question), params={"Retriever": {"top_k": 10}})
    
    print("documents retrieved")
    
    l=[]
    
    for x in res['documents']:
        
        row = df[df['title']==x.meta['name']]
        obj = Movie()
        obj.title = row['title'].to_list()[0]
        obj.summary = row['summary'].to_list()[0]
        obj.rating = row['rating'].to_list()[0]
        obj.runtime = row['runtime'].to_list()[0]
        obj.year = row['year'].to_list()[0]
        obj.image = "https://www." + str(row['image_720p'].to_list()[0])
        obj.cast = row['cast'].to_list()[0]
        obj.director = row['director'].to_list()[0]
        obj.genre = " | ".join(row['genre'].to_list()[0].split(','))
        obj.certificate = row['certificate'].to_list()[0]
        obj.modal_id = "".join(row['title'].to_list()[0].split())
        obj.modal_cast = " | ".join(row['cast'].to_list()[0])
        l.append(obj)
    
    x = len(l)%4
    l2 = np.array(l[:-x]).reshape(-1,4).tolist()
    l2.append(l[-x:])
    
    movie_list = l2
    
    return movie_list

def search_by_title(title):
    
    df = MovieAppConfig.data
    query_vec = MovieAppConfig.tr.encode(title)
    similarity = cosine_similarity(np.array(query_vec).reshape(1,-1),np.array(df['t_vec'].tolist())).flatten()
    
    indices = (list(enumerate(similarity)))
    
    final_indices = list(filter(filter_indices1,indices))
    
    l=[]
        
    for r in final_indices:
        
        row = df.iloc[r[0]]
        obj = Movie()
        obj.title = row['title']
        obj.summary = row['summary']
        obj.rating = row['rating']
        obj.runtime = row['runtime']
        obj.year = row['year']
        obj.image = "https://www." + str(row['image_720p'])
        obj.cast = row['cast']
        obj.director = row['director']
        obj.genre = " | ".join(row['genre'].split(','))
        obj.certificate = row['certificate']
        obj.modal_id = "".join(row['title'].split())
        obj.modal_cast = " | ".join(row['cast'])
        l.append(obj)
        
    x = len(l)%4
    l2 = np.array(l[:-x]).reshape(-1,4).tolist()
    l2.append(l[-x:])
    
    movie_list = l2
    
    print(f"Movie List Created of length {len(movie_list)}")
        
    return movie_list

def search_by_cast(cast):
    
    print(f"Called search_by_cast function for {cast}")
    
    df = MovieAppConfig.data
    
    model = MovieAppConfig.tr
    
    retriever = MovieAppConfig.retriever
    
    doc_store = MovieAppConfig.document_store
    
    query_vec = model.encode(cast)
    
    similarity1 = cosine_similarity(np.array(query_vec).reshape(1,-1),np.array(df['d_vec'].tolist())).flatten()
    similarity2 = cosine_similarity(np.array(query_vec).reshape(1,-1),np.array(df['c_vec'].tolist())).flatten()
    
    indices1 = (list(enumerate(similarity1)))
    indices2 = (list(enumerate(similarity2)))
    
    final_indices1 = list(filter(filter_indices1,indices1))
    final_indices2 = list(filter(filter_indices2,indices2))
    final_indices2.sort(key=lambda x:x[1], reverse=True)
    
    if(len(final_indices1)==0):
        final_indices1.extend(final_indices2)
    
    l=[]
        
    for r in final_indices1:
        
        row = df.iloc[r[0]]
        obj = Movie()
        obj.title = row['title']
        obj.summary = row['summary']
        obj.rating = row['rating']
        obj.runtime = row['runtime']
        obj.year = row['year']
        obj.image = "https://www." + str(row['image_720p'])
        obj.cast = row['cast']
        obj.director = row['director']
        obj.genre = " | ".join(row['genre'].split(','))
        obj.certificate = row['certificate']
        obj.modal_id = "".join(row['title'].split())
        obj.modal_cast = " | ".join(row['cast'])
        l.append(obj)
        
    x = len(l)%4
    l2 = np.array(l[:-x]).reshape(-1,4).tolist()
    l2.append(l[-x:])
    
    movie_list = l2
    
    print(f"Movie List Created of length {len(movie_list)}")
        
    return movie_list
        
    
def search(request):
    
    df = MovieAppConfig.data
    
    query = str(request.GET['search_movie'])
    
    query_list = []
    
    query_list.append(query)
    
    prediction = MovieAppConfig.predictor.predict(query_list).argmax(axis=-1)
    
    classes = ['cast','synopsis','title']
    
    intent = classes[prediction[0]]
    
    if(intent == 'cast'):
        
        print("Intent cast recognized")
        
        movie_list = search_by_cast(query)
    
    elif(intent == 'title'):
        
        print("Intent title recognized")
        
        movie_list = search_by_title(query)
        
    elif(intent == 'synopsis'):
        
        print("inten synopsis recognized")
        
        movie_list = search_by_synopsis(query)
        
    return render(request,'search.html',{'movies_list':movie_list})
        

# Create your views here.
def index(request):

    
    df = MovieAppConfig.data
    
    model = MovieAppConfig.predictor
    
    df_sorted = df.sort_values(by=['year'],ascending=False)[:12]
    
    movies_list = []
    
    k=0
    
    for i in range(3):
        l=[]
        for j in range(4):
            
            obj = Movie()
            obj.title = df_sorted.iloc[k]['title']
            obj.summary = df_sorted.iloc[k]['summary']
            obj.rating = df_sorted.iloc[k]['rating']
            obj.runtime = df_sorted.iloc[k]['runtime']
            obj.year = df_sorted.iloc[k]['year']
            obj.image = "https://www." + str(df_sorted.iloc[k]['image_720p'])
            obj.cast = df_sorted.iloc[k]['cast']
            obj.director = df_sorted.iloc[k]['director']
            obj.genre = " | ".join(df_sorted.iloc[k]['genre'].split(','))
            obj.certificate = df_sorted.iloc[k]['certificate']
            obj.modal_id = "".join(df_sorted.iloc[k]['title'].split())
            obj.modal_cast = " | ".join(df_sorted.iloc[k]['cast'])
            
            k+=1
            
            l.append(obj)
            
        movies_list.append(l)
    
    
    return render(request, "index.html", {'movies_list': movies_list})