# -*- coding: utf-8 -*-
import scrapy


class CsrhubRatingsSpider(scrapy.Spider):
    name = 'csrhub_ratings'
    allowed_domains = ['www.csrhub.com/CSR_and_sustainability_information']
    start_urls = ['https://www.csrhub.com/CSR_and_sustainability_information/Orange-SA']

    def parse(self, response):
        company_section = response.xpath('//div[@class="company-section_top-rank_num"]/span/span[@class="value"]/text()')
        company_section.xpath("./")
