# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

# scrapy genspider -t crawl news  www.esgtoday.com/category/esg-news/companies

class NewsSpider(CrawlSpider):
    name = 'news'
    allowed_domains = ['www.esgtoday.com/category/esg-news/companies']
    start_urls = ['http://www.esgtoday.com/category/esg-news/companies/']

    rules = (
        Rule(LinkExtractor(allow=r'Items/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        item = {}
        #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        #item['name'] = response.xpath('//div[@id="name"]').get()
        #item['description'] = response.xpath('//div[@id="description"]').get()
        return item
