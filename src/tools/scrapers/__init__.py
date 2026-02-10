"""Web scraping tools"""

from .amazon_product_scraper import AmazonProductScraper
from .amazon_scraper import AmazonScraper
from .deals_scraper import AmazonDealsScraper

__all__ = [
    "AmazonScraper",
    "AmazonProductScraper",
    "AmazonDealsScraper",
]
