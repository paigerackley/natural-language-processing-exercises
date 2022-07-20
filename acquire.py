from requests import get
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import os



def get_blog_urls():
    headers = {'user-agent': 'Innis Codeup Data Science'}
    response = get('https://codeup.com/blog/', headers=headers)
    soup = BeautifulSoup(response.text, features="lxml")
    urls = [a.attrs['href'] for a in soup.select('a.more-link')]
    return urls
    

def parse_blog(url):
    url = url
    #establish header
    headers = {'User-Agent': 'Codeup Data Science'}
    response = get(url, headers=headers)
    
    # Make a soup variable holding the response content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    output = {}
    output['title'] = soup.find('h1', class_='entry-title').text
    output['date'] = soup.find('span', class_='published').text
    output['category'] = soup.find('a', rel = 'category tag').text
    output['content'] = soup.find('div', class_= 'entry-content').text.strip().replace('\n', ' ').replace('\xa0', ' ')
    
    return output



def get_blog_articles(use_cache=True):
    
    if os.path.exists('blog_articles.json') and use_cache:
        return pd.read_json('blog_articles.json')

    urls = get_blog_urls()
    output = []
    
    for url in urls:
        output.append(parse_blog(url))
        
    df = pd.DataFrame(output)
    df.to_json('blog_articles.json', orient='records')
    
    return df


def parse_news_article(article, category):
    output = {}

    output['category'] = category
    output['title'] = article.find('span', itemprop = 'headline').text.strip()
    output['author'] = article.find('span', class_ = 'author').text
    output['date'] = article.find('span', clas = 'date').text.split(',')[0]
    output['content'] = article.find('div', itemprop = 'articleBody').text

    return output



def parse_news_page(category):
    url = 'https://inshorts.com/en/read/' + category
    response = get(url)
    soup = BeautifulSoup(response.text)

    cards = soup.select('.news-card')
    articles = []

    for card in cards:
        articles.append(parse_news_article(card, category))

    return articles



# cache the data, and turn it into a dataframe (function)
def get_news_articles(use_cache=True):
    if os.path.exists('news_articles.json') and use_cache:
        return pd.read_json('news_articles.json')

    categories = ['business', 'sports', 'technology', 'entertainment']

    articles = []

    for category in categories:
        articles.extend(parse_news_page(category))

    df = pd.DataFrame(articles)
    df.to_json('news_articles.json', orient='records')
    return df