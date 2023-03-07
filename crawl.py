from time import sleep
import datetime as dt
from newspaper import Article
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from nltk.corpus import stopwords
from http.cookiejar import CookieJar as cj
from urllib.parse import urlparse
from tqdm import tqdm
import os
import gc
import psutil
import json


class SearchEngine(object):
    def __init__(self,
                 name='CNN',
                 keyword='US',
                 startpage=1,
                 endpage=2,
                 sleep1=0.5,
                 sleep2=0.5,
                 sleep3=0.5,
                 filter_=None,
                 process=True,
                 parse=False,
                 save=False,
                 root='data',
                 limit1=10,
                 limit2=10,
                 limit3=1000,):
        '''
        This class is responsible for getting news urls, parsing them, and saving them
         from different news websites and keywords.

        Parameters
        ----------
        name: str, name of the news website
        keyword: str, keyword to search
        startpage: int, the start page of the search
        endpage: int, the end page of the search
        sleep1: float, time interval to revisit the page if failed
        sleep2: float, time interval to visit the next page if current page is empty
        sleep3: float, time interval to reparse the page if failed
        limit1: int, number of times to revisit the page if failed, then skip the page
        limit2: int, number of times to visit next page if current page is empty, then stop the crawler
        limit3: int, total number of times to reparse the page if failed, then stop the crawler
        filter_: dict, time filter for the news
        process: bool, whether to get the urls in debug
        parse: bool, whether to parse the urls in debug
        save: bool, whether to save the news in debug
        root: str, root directory to save the news

        kw: dict, method to process the keywords for different news websites
        headers: dict, headers for the requests
        page: int, current page
        methods: dict, methods to get the urls for different news websites
        method: str, method to get the urls indexed from methods
        info: dict, search page urls for different news websites
        loc: dict, location of the news urls for different search pages
        miss_domain: list, news websites that do not have the domain in the news urls
        has_domain: bool, whether the news website has the domain in the news urls
        easy_json: bool, whether the news website has easy json format
        medium_json: bool, whether the news website has medium json format
        hard_json: bool, whether the news website has hard json format
        jsonids: dict, location of the news urls for different json formats
        urls: list, news urls
        domain: str, domain of the news website
        news: list of News object
        texts: list, texts of the news
        titles: list, titles of the news
        publish_dates: list, publish dates of the news
        mem: psutil.virtual_memory(), memory usage
        num: int, number of news
        count: int, number of news urls parsed
        driver: webdriver, driver for selenium
        pre: str, location of the parent node of the news urls
        path: str, location of the news urls to be saved
        '''
        self.name = name
        self.kw = {'CNN': '+',
                   'foxnews': '%20',
                   'time': '+',
                   'ABC': '%2520',
                   'spectator': '%20',
                   'blaze': '%2B',
                   'dailycaller': '%20',
                   'federalist': '+',
                   'nypost': '+'}
        self.keyword = self.kw[self.name].join(keyword.split())
        self.headers = {'headers': {'User-Agent': 'newspaper/0.2.8'},
                        'cookies': cj(),
                        'timeout': 7,
                        'allow_redirects': True,
                        'proxies': {}}
        if filter_:
            self.filter_ = filter_.copy()
            self.process_filter()
        else:
            self.filter_ = {'begin_time': '', 'end_time': ''}
        self.page = startpage
        self.startpage = startpage
        self.endpage = endpage
        self.methods = {'direct': ['time', 'spectator', 'federalist', 'nypost'],
                        'api': ['foxnews', 'CNN', 'blaze', 'dailycaller'],
                        's': ['ABC']}
        self.set_method()
        if self.name in self.methods['s']:
            self.method = 's'
        self.info = None
        self.loc = {'time': 'media-img margin-8-bottom',
                    'foxnews': 'link',
                    'CNN': 'url',
                    'ABC': 'AnchorLink',
                    'spectator': '',
                    'blaze': 'widget__headline-text custom-post-headline',
                    'dailycaller': 'clicktrackUrl',
                    'federalist': 'd-block position-relative mb-20',
                    'nypost': 'postid'}
        self.miss_domain = ['nytimes', 'nationalreview']
        self.has_domain = self.name not in self.miss_domain
        self.easy_json = self.name in ['foxnews', 'dailycaller', 'nationalreview']
        # self.easy_jsons = {'foxnews': 'link'}
        self.medium_json = self.name in ['blaze']
        self.hard_json = self.name in ['CNN']
        self.jsonids = {'CNN': ['result'],
                      'blaze': ['posts_html'],
                      'foxnews': ['items'],
                      'dailycaller': ['results'],
                      'nationalreview': ['template_items', 'items']}
        self.urls = []
        self.sleep1 = sleep1
        self.sleep2 = sleep2
        self.sleep3 = sleep3
        self.domain = None
        self.news = []
        self.texts = []
        self.titles = []
        self.publish_dates = []
        self.root = root
        self.count = 0
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        self.limit1 = limit1
        self.limit2 = limit2
        self.limit3 = limit3
        self.mem = psutil.virtual_memory()
        self.num = 0
        # Here we do not use the webdriver as it is slow, but you can use it if you want
        # options = webdriver.ChromeOptions()
        # options.add_argument('--ignore-certificate-errors')
        # options.add_argument('--incognito')
        # options.add_argument('--headless')
        # self.driver = webdriver.Chrome("chromedriver_mac64/chromedriver", options=options)
        self.driver = None
        self.pre = {'ABC': 'Search__body__wrapper w-100',
                    'spectator': 'main-loop'}
        # Here we do not use the filter, so we do not need to add the filter to the path, but you can add it if you want
        # self.path = self.root + f'/{self.name}_{self.keyword}_page{self.startpage}to{self.endpage}_time
        # {self.filter_["begin_time"]}to{self.filter_["end_time"]}.csv'
        self.path = self.root + f'/{self.name}/{self.name}_{self.keyword}.csv'
        if process:
            self.get_all_urls()
        if parse:
            self.parse()
        if save:
            self.save()

    def process_filter(self):
        '''
        Process the time filter, which is used to handle time requirements.
        Sometimes we need news from a certain period of time,
        so we use this filter to standardize the time list we input into a format
        that can be put into the URL.
        '''
        if self.filter_['begin_time']:
            self.filter_['begin_time'] = dt.datetime(*self.filter_['begin_time']).strftime('%Y%m%d')
        if self.filter_['end_time']:
            self.filter_['end_time'] = dt.datetime(*self.filter_['end_time']).strftime('%Y%m%d')
        return None

    def set_method(self):
        '''
        Set the method of getting the news urls
        '''
        if self.name in self.methods['direct']:
            self.method = 'direct'
        elif self.name in self.methods['api']:
            self.method = 'api'
        elif self.name in self.methods['s']:
            self.method = 's'

    def get_urls(self, page):
        '''
        Get the news urls of a search page
        It is the core part. It is responsible for accessing a search page
        and extracting and processing the news URLs from the page.
        This method uses self.info to obtain the corresponding search page.
        Each time this method is called, a self.page is defined,
        and then the URLs of a certain page can be obtained from self.info.
        '''
        self.page = page
        print(f'Getting page {self.page} of {self.endpage} from {self.name}...', end='\r', flush=True)

        # define the url of the search page so that we can get the news urls from a specific search page
        self.info = {'time': f'https://time.com/search/?q={self.keyword}&page={self.page}',
                     'foxnews': f"https://api.foxnews.com/search/web?q={self.keyword}"
                                f"+-filetype:amp+-filetype:xml+more:pagemap:metatags-prism.section+"
                                f"more:pagemap:metatags-pagetype:article+more:pagemap:metatags-dc.type:"
                                f"Text.Article&siteSearch=foxnews.com&siteSearchFilter=i&sort=date:r:"
                                f"{self.filter_['begin_time']}:{self.filter_['end_time']}&start={self.page-1}"
                                f"1&callback=__jp5",
                     'CNN': f'https://search.api.cnn.com/content?q={self.keyword}&size=10&from='
                            f'{10*self.page-10}&page={self.page}&sort=relevance&types=article',
                     "ABC": f'https://abcnews.go.com/search?searchtext={self.keyword}&type=Story&page={self.page}',
                     'spectator': f'https://spectator.org/page/{self.page}/?s={self.keyword}',
                     'blaze': f'https://www.theblaze.com/res/load_more_posts/data.js?site_id=19257436&node'
                              f'_id=%2Froot%2Fblocks%2Fblock%5Bsearch%5D%2Fabtests%2Fabtest%5B1%5D%2'
                              f'Felement_wrapper%2Fchoose%2Fotherwise%2Felement_wrapper%5B2%5D%2'
                              f'Felement_wrapper%5B2%5D%2Fchoose%2Fotherwise%2Fposts-&resource'
                              f'_id=search_US+good&path_params=%7B%7D&formats=html&q={self.keyword}'
                              f'&rm_lazy_load=1&exclude_post_ids=&pn={self.page}&pn_strategy=',
                     'dailycaller': f'https://cse.google.com/cse/element/v1?rsz=filtered_cse&'
                                    f'num=10&hl=en&source=gcsc&gss=.com&start={self.page*10}'
                                    f'&cselibv=c23214b953e32f29&cx=013858372769713515008:m9uq4uupsfm'
                                    f'&q={self.keyword}&safe=off&cse_tok=ALwrddFDewNY08F8bYp5sX7stOM'
                                    f'4:1677412988743&exp=csqr,cc&callback=google.search.cse.api3335',
                     'federalist': f'https://thefederalist.com/page/{self.page}/?s={self.keyword}',
                     'nypost': f'https://nypost.com/search/{self.keyword}/page/{self.page}/',}
        self.domain = urlparse(self.info[self.name]).netloc

        # get the news urls from the search page with different methods
        if self.method == 'direct':
            soup = BeautifulSoup(requests.get(self.info[self.name], **self.headers).text, 'lxml')
            if self.name in self.pre:
                soup = soup.find('div', class_=self.pre[self.name])
            urls = [tag['href'] for tag in soup.select(f'a[class*="{self.loc[self.name]}"]')
                    if tag.has_attr('href')]
        elif self.method == 'api':
            js = self.get_dict(self.get_json(requests.get(self.info[self.name], **self.headers).text))
            if self.easy_json:
                urls = list(map(lambda x: x[self.loc[self.name]], js))
            elif self.medium_json:
                urls = [tag['href'] for tag in BeautifulSoup(js, 'lxml')
                .find_all('a', class_=self.loc[self.name]) if tag.has_attr('href')]
            elif self.hard_json:
                urls = [i['url'] for i in js if 'url' in i]
            # else:
            #     text = np.array(requests.get(self.info[self.name], **self.headers).text.split())
            #     urls = list(map(lambda x: x.strip('",'), text[np.where(text == '"link":')[0] + 1]))
        elif self.method == 's':
            options = webdriver.ChromeOptions()
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--incognito')
            options.add_argument('--headless')
            self.driver = webdriver.Chrome("chromedriver_mac64/chromedriver", options=options)
            self.driver.get(self.info[self.name])
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            if self.name in self.pre:
                soup = soup.find('div', class_=self.pre[self.name])
            urls = [tag['href'] for tag in soup.find_all('a', class_=self.loc[self.name]) if tag.has_attr('href')]

        # store the urls in a list and add the domain if needed
        try:
            return sum(self.add_domain(urls), [])
        except:
            return self.add_domain(urls)

    def add_domain(self, urls):
        '''
        add the domain to the urls if needed
        '''
        if not self.has_domain:
            return list(map(lambda x: f'https://{self.domain}{x}', urls))
        return urls

    def get_json(self, string):
        '''
        get the json dictionary from the string
        '''
        return json.loads(string[string.index('{'):string.rindex('}')+1])

    def get_dict(self, dic):
        '''
        extract a small dictionary containing the news urls from the json dictionary.
        '''
        for i in self.jsonids[self.name]:
            dic = dic[i]
        return dic

    def get_all_urls(self):
        '''
        Get all the news urls by repeatedly calling get_urls()
        '''
        urls = []
        while self.page <= self.endpage:
            try:
                urls = self.get_urls(self.page)
                self.page += 1
                self.count1 = 0
                self.count2 = 0
            except:
                sleep(self.sleep1)
                self.count1 += 1
                # stop when failed for too many consecutive times
                if self.count1 >= self.limit1:
                    print(f'Getting urls No.{self.page} from {self.keyword} failed too many times!', flush=True)
                    break
                continue

            # stop when the urls are empty for too many consecutive times
            if urls == []:
                sleep(self.sleep2)
                self.count2 += 1
                self.page += 1
                if self.count2 >= self.limit2:
                    print(f'Getting pages from {self.keyword} empty too many times!', flush=True)
                    break

            self.urls.extend(urls)

        self.urls = sorted(list(set(self.urls)))

    def parse(self):
        '''
        Get and parse the news from the news urls to get the titles, texts and publish dates
        by using the News class.
        '''
        print(f'Parsing {self.num} urls from {self.name}...', flush=True)

        for url in tqdm(self.urls, colour='green'):
            try:
                new = News(url)
                self.count += 1
                self.texts.append(new.text)
                self.titles.append(new.title)
                self.publish_dates.append(new.publish_date)
            except:
                sleep(self.sleep3)
                self.count3 += 1
                print(f'Failed to parse No. {self.count} of {len(self.urls)} urls from {self.name}, '
                      f'total failed {self.count3} times.', flush=True)
                # stop when failed for too many times
                if self.count3 >= self.limit3:
                    print(f'Parsing {self.keyword} failed too many times!', flush=True)
                    print(f'{self.keyword} saved {self.count} results', flush=True)
                    break

            # save the results every 100 urls, then clean the memory
            if self.count % 100 == 0:
                self.save()
                num = len(self.texts)
                self.urls = self.urls[num:]
                self.titles = []
                self.texts = []
                self.publish_dates = []
                gc.collect()
                self.get_system_memory()
                print('-'*30, flush=True)

        return None

    def get_system_memory(self):
        '''
        get the system memory usage information
        '''
        self.mem = psutil.virtual_memory()
        print(f'Total memory: {self.mem.total/1024/1024/1024:.2f}GB, '
              f'Used memory: {self.mem.used/1024/1024/1024:.2f}GB, '
              f'Available memory: {self.mem.available/1024/1024/1024:.2f}GB',
              flush=True)

    def save(self):
        '''
        save the results to a csv file
        '''
        # self.path = self.root + f'/{self.name}_{self.keyword}_page{self.startpage}to{self.endpage}' \
        #                         f'_time{self.filter_["begin_time"]}to{self.filter_["end_time"]}.csv'
        self.path = self.root + f'/{self.name}/{self.name}_{self.keyword}.csv'

        df = pd.DataFrame({'title': self.titles,
                           'text': self.texts,
                           'url': self.urls[:len(self.texts)],
                           'published_time': self.publish_dates})
        # use mode='a' to append the data to the csv file
        df.to_csv(self.path, mode='a', header=not os.path.exists(self.path), index=False)
        print(f'Saved to {self.path}', flush=True)

    def remove_dupna(self):
        '''
        remove the duplicate and na rows in the csv file
        '''
        pd.read_csv(self.path).drop_duplicates().dropna().to_csv(self.path, index=False)
        print(f'Removed duplicate and na rows from {self.path}', flush=True)

    def go(self):
        '''
        the main function to run the whole process to get the news of the keyword from the media
        '''
        self.get_all_urls()
        self.num = len(self.urls)
        self.parse()
        self.save()
        return None

    def init(self):
        '''
        Initialize the parameters for another keyword or another media
        '''
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        self.count = 0
        self.urls = []
        self.news = []
        self.titles = []
        self.texts = []
        self.publish_dates = []
        self.page = self.startpage
        self.set_method()
        gc.collect()

    def auto(self, medias, keywords):
        '''
        Automatically get the news of the keywords from the medias
        '''
        print(f'Reminder: you can only stop this process by restarting the kernel, '
              f'or double click the stop button in some cases', flush=True)

        for media in medias:
            for keyword in keywords:
                print('-'*30, flush=True)
                print(f'Keywords: {keywords}', flush=True)
                print(f'Current keyword: {keyword}', flush=True)
                self.name = media
                self.keyword = self.kw[self.name].join(keyword.split())
                self.init()
                self.go()
                self.remove_dupna()
                print(f'{self.keyword} done', flush=True)
                print('\n', flush=True)


class News(Article):
    '''
    the class to parse the news from the url to get the title, text and publish date.
    it inherits the class Article from newspaper3k

    Parameters
    ----------
    url: str, the url of the news
    publish_date: str, the published date of the news
    title: str, the title of the news
    text: str, the body text of the news
    '''
    def __init__(self, url):
        super().__init__(url)
        self.download()
        self.parse()
        try:
            self.publish_date = self.publish_date.strftime('%Y-%m-%d')
        except:
            self.publish_date = 'N/A'
        if not self.text:
            self.text = 'N/A'
        if not self.title:
            self.title = 'N/A'