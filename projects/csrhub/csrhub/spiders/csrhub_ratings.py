# -*- coding: utf-8 -*-
import scrapy


class CsrhubRatingsSpider(scrapy.Spider):
    name = 'csrhub_ratings'
    allowed_domains = ['www.csrhub.com/CSR_and_sustainability_information']
    start_urls = [
        'https://www.csrhub.com/CSR_and_sustainability_information/Orange-SA',
        'https://www.csrhub.com/CSR_and_sustainability_information/Michelin'
    ]

    def parse(self, response):
        # company_section = response.xpath('//div[@class="company-section_top-rank_num"]/span/span[@class="value"]/text()')
        # company_section.xpath("./")
        issues = response.xpath('//ul[@class="company-section_spec-issue_list"]/li/img/@title').getall()
        description = response.xpath('//div[@class="company-section_descr"]/p/text()').getall()
        rows = response.xpath('//div[@class="company-section_sheet"]/table/tr')
        ticker = rows[0].xpath('.//td[2]/text()').get()
        isin = rows[1].xpath('.//td[2]/text()').get()
        #address = rows.xpath(".//tr[3]/td[2]/text()").get()
        website = rows[4].xpath('.//td[2]/a/@href').get()
        phone = rows[5].xpath('.//td[2]/text()').get()
        industry = rows[7].xpath('.//td[2]/a/@href').get()
        yield {
            "ticker": ticker,
            "isin": isin,
            "website": website,
            "phone": phone,
            "industry": industry,
            "company_issues": issues,
            "company_description": description
            }

