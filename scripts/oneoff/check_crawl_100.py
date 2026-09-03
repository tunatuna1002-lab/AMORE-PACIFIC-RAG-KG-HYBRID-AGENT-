"""크롤러 수정 후 100개 수집 검증 테스트 (Railway에서 실행)"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.scrapers.amazon_scraper import AmazonScraper  # noqa: E402


async def test_single_category():
    scraper = AmazonScraper()
    await scraper.initialize()

    # Lip Care 카테고리 1개만 테스트
    url = "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care/zgbs/beauty/3761351"
    result = await scraper.scrape_category("lip_care", url)

    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    print(f"Total products: {result['count']}")

    if result["products"]:
        ranks = [p["rank"] for p in result["products"]]
        print(f"Rank range: {min(ranks)} ~ {max(ranks)}")
        print(f"First 5 ranks: {ranks[:5]}")
        print(f"Last 5 ranks: {ranks[-5:]}")

        # 누락 확인
        expected = set(range(1, 101))
        actual = set(ranks)
        missing = expected - actual
        if missing:
            print(f"Missing ranks: {sorted(missing)}")
        else:
            print("All 100 ranks present!")

    await scraper.close()


asyncio.run(test_single_category())
