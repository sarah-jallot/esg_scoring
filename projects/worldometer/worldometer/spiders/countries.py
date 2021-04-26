# -*- coding: utf-8 -*-
import scrapy
from logzero import logger
from scrapy.shell import inspect_response
from scrapy.utils.response import open_in_browser

class CountriesSpider(scrapy.Spider):
    name = 'countries'
    allowed_domains = ['www.worldometers.info']
    start_urls = ['https://www.worldometers.info/world-population/population-by-country/']

    #def start_requests(self):
    #    yield scrapy.Request(
    #        url="https://www.worldometers.info/world-population/population-by-country/",
    #        callback= self.parse,
    #        headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36"}
    #    )
    def parse(self, response):
        logger.warning(response.status)
        countries = response.xpath("//td/a")

        for country in countries:
            name = country.xpath(".//text()").get()
            link = country.xpath(".//@href").get()
            print(link)
            # absolute_url = f"https://worldometers.info{link}"
            # absolute_url = response.urljoin(link)
        # yield scrapy.Request(url=absolute_url)
            yield response.follow(
                url=link,
                callback=self.parse_country,
                meta={"country_name":name}) # to follow a relative url !

    def parse_country(self, response):
       # inspect_response(response, self)
       # open_in_browser(response, self)
        logger.debug("hello")
        logger.info("info")
        logger.warning("warn")
        logger.error("error")
        name = response.request.meta["country_name"]
        rows = response.xpath('(//table[@class="table table-striped table-bordered table-hover table-condensed table-list"])[1]/tbody')
        for row in rows:
            year = rows.xpath('.//td[1]/text()').get()
            population = row.xpath('.//td[2]/strong/text()').get()
            yield {
                "country_name": name,
                "country_year": year,
               "country_population": population
           }

# to debug : scrapy parse --spider=countries -c parse_country --meta='{\"country_name\":\"China\"}' https://www.worldometers.info/world-population/china-population/