"""Amazon 베스트셀러 페이지 3, 4 구조 디버그"""

import asyncio
import random

from playwright.async_api import async_playwright

try:
    from playwright_stealth import Stealth

    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False


async def debug_pages():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        page = await context.new_page()
        if HAS_STEALTH:
            s = Stealth()
            await s.apply_stealth_async(page)

        base = "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care/zgbs/beauty/3761351"

        for pg_num in [3, 4]:
            print(f"\n=== PAGE {pg_num} ===")
            await asyncio.sleep(random.uniform(5, 8))
            url = f"{base}?pg={pg_num}"
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(5)
            await page.evaluate("window.scrollBy(0, 500)")
            await asyncio.sleep(2)

            content = await page.content()
            blocked = any(
                x in content
                for x in [
                    "Enter the characters you see below",
                    "just need to make sure you're not a robot",
                ]
            )
            if blocked:
                print(f"PAGE {pg_num}: BLOCKED!")
                continue

            cards = await page.query_selector_all('[data-asin]:not([data-asin=""])')
            print(f"[data-asin] cards: {len(cards)}")

            badges = await page.query_selector_all("span.zg-bdg-text")
            if badges:
                first3 = [await b.inner_text() for b in badges[:3]]
                last3 = [await b.inner_text() for b in badges[-3:]]
                all_ranks = [(await b.inner_text()).strip().replace("#", "") for b in badges]
                print(f"Ranks: {len(badges)}, first={first3}, last={last3}")
                print(f"All ranks: {all_ranks}")
            else:
                print("No rank badges found")

            # Check pagination links
            pg_links = await page.query_selector_all('a[href*="pg="]')
            for link in pg_links:
                href = await link.get_attribute("href") or ""
                text = (await link.inner_text()).strip()
                if text in ["1", "2", "3", "4", "5", "Next page", "Previous page"]:
                    print(f"  Pagination: text='{text}' href=...{href[-50:]}")

        await browser.close()
        print("\n=== DONE ===")


asyncio.run(debug_pages())
