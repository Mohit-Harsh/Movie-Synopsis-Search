# Movie-Synopsis-Search

## About The Project


https://github.com/Mohit-Harsh/Movie-Synopsis-Search/assets/111693866/bb64e6b3-06c5-44c5-b115-9e7a59fcc69b



A Django Application in which you can search a Movie by it's Title, Cast, Director or Synopsis.


### Built with


  * Tensorflow
  * BeautifulSoup
  * Sentence Transformers from Huggingface
  * Dense Passage Retriever from Haystack
  * Django



## Getting Started

### Step 1: Web Scraping

* First we scraped the data from ```IMDB``` website using ```BeautifulSoup```
* The Notebook related to ```IMDB Web Scraping``` can be found here - https://www.kaggle.com/code/pmohitharsh/imdb-movies-data-scrape/notebook
* The Movies Dataset can be found here - https://www.kaggle.com/datasets/pmohitharsh/imdb-movies-synopsis

### Step 2: Data Cleaning

* After scraping we cleaned the data by converting ```Year```, ```Rating``` and ```Runtime``` columns to ```Integer``` type from ```String```.
* Then we removed some unnecessary symbols that came along with the text while scraping.


### Step 3: Vector Embeddings


* In this step we created vector embeddings for ```Title```, ```Director``` and ```Cast``` columns using Sentence Transformers.
* The Notebook related to Sentence Embeddings can be found here - 
* We also created vector embeddings for the ```Synopsis``` using ```Facebook Context Encoders``` and ```Haystack DPR``` module.
* We saved the above embeddings as documents in ```FAISS Document Store``` from Haystack.
* The Notebook related to Context Embeddings can be found here - https://www.kaggle.com/code/pmohitharsh/imdb-dpr-training
