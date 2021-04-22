# -*- coding: utf-8 -*-
import scrapy


class RatingsSpider(scrapy.Spider):
    name = 'ratings'
    allowed_domains = ['www.msci.com/our-solutions/esg-investing/esg-ratings/esg-ratings-corporate-search-tool/issuer/orange-sa/IID000000002133302']
    start_urls = ['http://www.msci.com/our-solutions/esg-investing/esg-ratings/esg-ratings-corporate-search-tool/issuer/orange-sa/IID000000002133302/']

    def parse(self, response):
        pass
