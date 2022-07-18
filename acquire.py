from requests import get
from bs4 import BeautifulSoup
import os
import pandas as pd


#######################################################
# This file is for the acquire exercises for NLP codeup module.
#######################################################

def get_blog_articles(url):
    """this function pulls codeup blog urls and reassigns the tile and content of the blog
    into a dicitionary"""
    url = url
    #establishing a header for access:
    headers = {'User-Agent': 'Codeup Data Science Student'} 
    response = get(url, headers=headers)
    
    # Make a soup variable holding the response content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    #creating the dictionary
    output = {}
    output['title'] = soup.find('h1', class_='entry-title').text
    output['content'] = soup.find('div', class_='entry-content').text.strip().replace('\n',' ')
    
    return output

def parse_news_article(article, category):
    """this function pulls inshorts news articles and reassigns the tile and content of the articles
    by categoryinto a dicitionary"""
    
     #creating the dictionary
    output = {}
    
    output['title'] = article.find("span", itemprop = "headline").text.strip()
    output['content'] = article.find("div", itemprop = "articleBody").text
    output['author'] = article.find("span", class_ = "author").text
    output['date'] = article.find("span", class_ = "date").text
    output['source'] = article.find("a", class_ = "source")
    output['category'] = category
    
    return output

#### Modifying the above to dictionary all articles into one page:

def parse_news_page(category):
    url = "https://inshorts.com/en/read/"+category
    response = get(url)
    soup = BeautifulSoup(response.text)
    
    cards = soup.select('.news-card')
    articles = []
    
    for card in cards:
        articles.append(parse_news_article(card, category))
        
    return articles



def get_news_articles(use_cache = True):
    """creating one function that grabs all categories and 
    puts them into one dictionary, while adding a cache of the data:"""
    if os.path.exists('news_articles.json') and use_cache:
        return pd.read_json('news_articles.json')
    
    categories = ['business', 'sports', 'technology', 'entertainment']
    
    articles = []
    
    for category in categories:
        print(f'Getting {category} articles')
        articles.extend(parse_news_page(category))
        
    df = pd.DataFrame(articles)
    df.to_json('news_articles.json', orient ='records')
    return df