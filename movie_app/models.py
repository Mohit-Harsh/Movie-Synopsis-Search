from django.db import models

# Create your models here.
class Movie:
    
    title : str
    summary : str
    rating : int
    year : int
    runtime : int
    director : str
    cast : str
    image : str
    genre : str
    certificate : str
    modal_id: str
    modal_cast : str