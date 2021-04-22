# -*- coding: utf-8 -*-
import scrapy


class CsrhubRatingsSpider(scrapy.Spider):
    name = 'csrhub_ratings'
    allowed_domains = ['www.csrhub.com/CSR_and_sustainability_information']
    start_urls = ['http://www.csrhub.com/CSR_and_sustainability_information/']

    def parse(self, response):
        pass
