from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()
scrapper_api_key = os.getenv("SCRAPPER_API_KEY")

def ws(name):
    params = {
        "engine" : "google_shopping",
        "q" : name,
        "api_key" : scrapper_api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results["shopping_results"]
    return(shopping_results[0:5])