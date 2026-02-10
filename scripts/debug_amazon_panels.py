"""Amazon 베스트셀러 페이지 좌/우 패널 구조 디버그"""

import asyncio
import random

from playwright.async_api import async_playwright

try:
    from playwright_stealth import Stealth

    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False


async def debug_panels():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = await context.new_page()
        if HAS_STEALTH:
            await Stealth().apply_stealth_async(page)

        url = "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care/zgbs/beauty/3761351"
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)

        # Full page scroll to load lazy content
        for i in range(10):
            await page.evaluate(f"window.scrollBy(0, {500 + i * 200})")
            await asyncio.sleep(0.5)
        await asyncio.sleep(3)

        # Check zg-left-col and zg-right-col
        left = await page.query_selector("#zg-left-col")
        right = await page.query_selector("#zg-right-col")
        print(f"#zg-left-col exists: {left is not None}")
        print(f"#zg-right-col exists: {right is not None}")

        if left:
            left_cards = await left.query_selector_all('[data-asin]:not([data-asin=""])')
            left_badges = await left.query_selector_all("span.zg-bdg-text")
            left_ranks = [(await b.inner_text()).strip() for b in left_badges]
            print(f"LEFT panel: {len(left_cards)} cards, ranks: {left_ranks}")

        if right:
            right_cards = await right.query_selector_all('[data-asin]:not([data-asin=""])')
            right_badges = await right.query_selector_all("span.zg-bdg-text")
            right_ranks = [(await b.inner_text()).strip() for b in right_badges]
            print(f"RIGHT panel: {len(right_cards)} cards, ranks: {right_ranks}")

        # Also check for #gridItemRoot pattern
        grid_roots = await page.query_selector_all('[id^="gridItemRoot"]')
        print(f"\n[id^=gridItemRoot] elements: {len(grid_roots)}")
        if grid_roots:
            for g in grid_roots[:3]:
                gid = await g.get_attribute("id")
                print(f"  First: {gid}")
            for g in grid_roots[-3:]:
                gid = await g.get_attribute("id")
                print(f"  Last: {gid}")

        # Check for the overall container structure
        containers = await page.query_selector_all("div._cDEzb_grid-column_2hIsc")
        print(f"\ndiv._cDEzb_grid-column_2hIsc: {len(containers)}")

        # Total product items after full scroll
        all_cards = await page.query_selector_all('[data-asin]:not([data-asin=""])')
        all_badges = await page.query_selector_all("span.zg-bdg-text")
        all_ranks = [(await b.inner_text()).strip() for b in all_badges]
        print(f"\nAfter full scroll - cards: {len(all_cards)}, badges: {len(all_badges)}")
        print(f"All ranks: {all_ranks}")

        # Try page 2 with full scroll too
        print("\n=== PAGE 2 with full scroll ===")
        await asyncio.sleep(random.uniform(5, 8))
        await page.goto(url + "?pg=2", wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)
        for i in range(10):
            await page.evaluate(f"window.scrollBy(0, {500 + i * 200})")
            await asyncio.sleep(0.5)
        await asyncio.sleep(3)

        p2_cards = await page.query_selector_all('[data-asin]:not([data-asin=""])')
        p2_badges = await page.query_selector_all("span.zg-bdg-text")
        p2_ranks = [(await b.inner_text()).strip() for b in p2_badges]
        print(f"Page 2 - cards: {len(p2_cards)}, badges: {len(p2_badges)}")
        print(f"Page 2 ranks: {p2_ranks}")

        if right:
            p2_right = await page.query_selector("#zg-right-col")
            if p2_right:
                p2r_badges = await p2_right.query_selector_all("span.zg-bdg-text")
                p2r_ranks = [(await b.inner_text()).strip() for b in p2r_badges]
                print(f"Page 2 RIGHT panel: {len(p2r_badges)} badges, ranks: {p2r_ranks}")

        await browser.close()
        print("\n=== DONE ===")


asyncio.run(debug_panels())
