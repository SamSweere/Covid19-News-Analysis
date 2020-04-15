import requests


## Lexis Uni

url = ("https://advance.lexis.com/api/document/collection/cases/id/3RJN-3D30-0039-446P-00000-00")
response = requests.get(url)
print(response)

for i in response.iter_lines():
    print(i)

response.content

a = response.iter_content().decode("UTF-8")
response_text = [i.decode("UTF-8") for i in response.iter_content()]

for i in response.iter_content():
    response_text.append(i)




# https://newsapi.org/
# API key: 0641d94d734641e5bf07c26c39ba57c7

url = ("https://newsapi.org/v2/everything?q=Covid19&apiKey=0641d94d734641e5bf07c26c39ba57c7")
response = requests.get(url).json()
print(response)

res = dict(response)
n_articles = len(res["articles"])
print(f"Number of articles: {n_articles}")
res["articles"][0]


## Google API
https://newsapi.org/s/google-news-api
# API key: 0641d94d734641e5bf07c26c39ba57c7



from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='0641d94d734641e5bf07c26c39ba57c7')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                          sources='bbc-news,the-verge',
                                          category='business',
                                          language='en',
                                          country='us')

# /v2/everything
# apparently one pate is 20 results
all_articles = newsapi.get_everything(q='Covid19',
                                      from_param='2018-04-11',
                                      to='2018-04-14',
                                      language='en',
                                      sort_by='relevancy',
                                      page=5)

len(all_articles["articles"])

# /v2/sources
sources = newsapi.get_sources()




"""
    Module used to collect news stories from different countries from a library database.
    Uses selenium to simulate user input and manually go through each news page.
    save the news article.
"""
import time
import codecs
import os
from math import floor
from requestium import Session
from numpy.random import normal

urls_b = [
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AKenya%21Kenya%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper',
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AEngland%7CNorthern%2BIreland%7CScotland%7CWales%21Multiple%2BCountries%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper%2Flanguage%3AEnglish%21English',
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AIndia%21India%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper',
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AChina%21China%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper'
]

urls_a = [
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3ANigeria%21Nigeria%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper',
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AUSA%21USA%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper%2Flanguage%3AEnglish%21English',
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AAustralia%21Australia%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper',
    'http://infoweb.newsbank.com.proxy.uchicago.edu/resources/search/nb?p=WORLDNEWS&t=country%3AEgypt%7CSaudi%2BArabia%7CUnited%2BArab%2BEmirates%21Multiple%2BCountries%2Fyear%3A2017%212017%2Fstp%3ANewspaper%21Newspaper'
]

def chill(amount):
    """
        Waits a random (normally distributed) amount of time, according to three buckets.
        args:
            amount: either of 'low', 'mild' or 'long' to indicate intended length of the wait
    """
    translator = {'low':1, 'mild':4, 'long':13}
    time.sleep(translator[amount] + normal(loc=2, scale=1.5))

def get_to_page(page_no, session):
    """
        Navigates to a specific page of search results.
        Requires the session to be currently in the first page of search results.
        args:
            page_no: desired result page number
            session: requestium user session
    """
    #assuming page 0 of search results
    if page_no < 10:
        session.driver.find_element_by_link_text(u"{}".format(page_no)).click()
        chill('mild')
    else:
        num_times = floor((page_no-10)/4) + 1
        num = 9
        for _ in range(num_times):
            session.driver.find_element_by_link_text(u"{}".format(num)).click()
            chill('mild')
            num += 4
        session.driver.find_element_by_link_text(u"{}".format(page_no)).click()
        chill('mild')

def get_news_from_url(target_url, max_articles, file_prefix, page):
    """
        Given a URL that already includes the filtering, performs a search and
        acquires max_articles articles
    """
    session = Session('./chromedriver', browser='chrome', default_timeout=15)
    session.driver.implicitly_wait(30)
    session.driver.get(target_url)
    chill('mild')
    session.driver.find_element_by_id("nbplatform-basic-search-val0").clear()
    chill('low')
    session.driver.find_element_by_id("nbplatform-basic-search-val0").send_keys("'climate change'")
    chill('low')
    session.driver.find_element_by_id("nbplatform-basic-search-submit").click()
    chill('mild')
    if page > 1:
        get_to_page(page, session)
    links = session.driver.find_elements_by_css_selector("a.nb-doc-link")
    print("{} links on the page".format(links))
    master_counter = 0
    while master_counter < max_articles:
        per_page = len(links)
        for i in range(per_page):
            session.driver.find_elements_by_css_selector("a.nb-doc-link")[i].click()
            chill('long')
            file_name = '{}_{}.html'.format(file_prefix, master_counter)
            complete_name = os.path.join('./data/', file_name)
            file_object = codecs.open(complete_name, "w", "utf-8")
            html = session.driver.page_source
            file_object.write(html)
            ## Save file here
            master_counter += 1
            session.driver.back()
            chill('mild')
        session.driver.find_element_by_link_text(u"next ›").click()


#############################CHANGE FILE PREFIX IN EVERY RUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#get_news_from_url(urls_a[0], 100, 'NG9_10', 41)
#get_news_from_url(urls_a[1], 200, 'US9_10', 81)
#get_news_from_url(urls_a[2], 100, 'AU10', 91)
get_news_from_url(urls_a[3], 100, 'ME10', 91)