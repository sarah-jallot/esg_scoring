# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from logzero import logger
# scrapy genspider -t crawl news  www.esgtoday.com/category/esg-news/companies

class NewsSpider(CrawlSpider):
    name = 'news'
    allowed_domains = ['www.esgtoday.com']
    start_urls = ['https://www.esgtoday.com/category/esg-news/companies/']

  #  user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'

    rules = (
        Rule(LinkExtractor(restrict_xpaths="//div[@class='post-content']/h2['post-title']/a"), callback='parse_article', follow=True,),# process_request="set_user_agent"),
  #      Rule(LinkExtractor(restrict_xpaths="//li[@class='next arrow'']/a"), follow=True)
    )

  #  def start_requests(self):
  #     yield scrapy.Request(url='https://www.esgtoday.com/category/esg-news/companies/',
  #                          headers={"User-Agent":self.user_agent})

   # def set_user_agent(self, request, spider):
   #     request.headers["User-Agent"] = self.user_agent
   #     return request

    def parse_article(self, response):
        print(f"HELLO {response.url}")
        yield {
            "article_title" : response.xpath("//h1[@class='entry-title']/text()").get(),
            "article_content": " ".join(response.xpath("//div[@class='post-wrap']/div/descendant::node()/text()").getall()),
        }
# wrap your xpath in normalize-space() within the "" to remove whitespaces!
# docker run -it -p 8050:8050 scrapinghub/splash
# to re-run, just go in docker dashboard and open the last session, then click on this link :http://0.0.0.0:8050/"
# Then enter your website url into this