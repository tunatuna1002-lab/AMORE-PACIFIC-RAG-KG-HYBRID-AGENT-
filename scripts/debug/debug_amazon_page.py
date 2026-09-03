"""Amazon 베스트셀러 페이지 구조 디버그 스크립트 (Railway에서 실행)"""

import asyncio
import random

from playwright.async_api import async_playwright

try:
    from playwright_stealth import stealth_async

    HAS_STEALTH_V1 = True
except ImportError:
    HAS_STEALTH_V1 = False

try:
    from playwright_stealth import Stealth

    HAS_STEALTH_V2 = True
except ImportError:
    HAS_STEALTH_V2 = False


async def debug_page():
    print(f"Stealth v1: {HAS_STEALTH_V1}, v2: {HAS_STEALTH_V2}")

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

        # Apply stealth
        if HAS_STEALTH_V1:
            await stealth_async(page)
            print("Applied stealth v1")
        elif HAS_STEALTH_V2:
            s = Stealth()
            await s.apply_stealth_async(page)
            print("Applied stealth v2")

        url = "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care/zgbs/beauty/3761351"

        # === PAGE 1 ===
        print("\n=== PAGE 1 ===")
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
            print("PAGE 1: BLOCKED!")
        else:
            # Count data-asin cards
            all_asin_cards = await page.query_selector_all('[data-asin]:not([data-asin=""])')
            print(f"[data-asin] cards total: {len(all_asin_cards)}")

            # Count grid items (actual product cards)
            grid_items = await page.query_selector_all("div.zg-grid-general-faceout")
            print(f".zg-grid-general-faceout: {len(grid_items)}")

            # Count rank badges
            badges = await page.query_selector_all(".zg-bdg-text")
            if badges:
                first3 = [await b.inner_text() for b in badges[:3]]
                last3 = [await b.inner_text() for b in badges[-3:]]
                print(f"Rank badges: {len(badges)}, first={first3}, last={last3}")
            else:
                print("No .zg-bdg-text badges found")

            # Try other badge selectors
            badges2 = await page.query_selector_all("span.zg-bdg-text")
            print(f"span.zg-bdg-text: {len(badges2)}")

            # Check for #zg-right-col (page 2 container in old layout)
            right_col = await page.query_selector("#zg-right-col")
            print(f"#zg-right-col exists: {right_col is not None}")

            # Check list items
            list_items = await page.query_selector_all(".a-list-item .zg-grid-general-faceout")
            print(f".a-list-item .zg-grid-general-faceout: {len(list_items)}")

            # Check for numbered rank spans
            rank_spans = await page.query_selector_all("span.zg-bdg-text")
            if rank_spans:
                all_ranks = []
                for r in rank_spans:
                    t = await r.inner_text()
                    all_ranks.append(t.strip().replace("#", ""))
                print(f"All rank numbers: {all_ranks}")

            # Check pagination
            pg_links = await page.query_selector_all('a[href*="pg="]')
            for link in pg_links:
                href = await link.get_attribute("href") or ""
                text = (await link.inner_text()).strip()
                print(f"  Pagination link: text='{text}' href=...{href[-60:]}")

            # Check for "See top 100" or pagination buttons
            all_links = await page.query_selector_all("a")
            for link in all_links:
                text = (await link.inner_text()).strip()
                if "100" in text or "next" in text.lower() or "page" in text.lower():
                    href = await link.get_attribute("href") or ""
                    print(f"  Interesting link: text='{text}' href=...{href[-80:]}")

        # === PAGE 2 ===
        print("\n=== PAGE 2 ===")
        await asyncio.sleep(random.uniform(5, 8))
        url2 = url + "?pg=2"
        await page.goto(url2, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)
        await page.evaluate("window.scrollBy(0, 500)")
        await asyncio.sleep(2)

        content2 = await page.content()
        blocked2 = any(
            x in content2
            for x in [
                "Enter the characters you see below",
                "just need to make sure you're not a robot",
            ]
        )
        if blocked2:
            print("PAGE 2: BLOCKED!")
        else:
            cards2 = await page.query_selector_all('[data-asin]:not([data-asin=""])')
            print(f"[data-asin] cards total: {len(cards2)}")

            grid2 = await page.query_selector_all("div.zg-grid-general-faceout")
            print(f".zg-grid-general-faceout: {len(grid2)}")

            badges2 = await page.query_selector_all(".zg-bdg-text")
            if badges2:
                first3 = [await b.inner_text() for b in badges2[:3]]
                last3 = [await b.inner_text() for b in badges2[-3:]]
                print(f"Rank badges: {len(badges2)}, first={first3}, last={last3}")
            else:
                print("No .zg-bdg-text badges found")

            rank_spans2 = await page.query_selector_all("span.zg-bdg-text")
            if rank_spans2:
                all_ranks2 = []
                for r in rank_spans2:
                    t = await r.inner_text()
                    all_ranks2.append(t.strip().replace("#", ""))
                print(f"All rank numbers: {all_ranks2}")

        await browser.close()
        print("\n=== DONE ===")


asyncio.run(debug_page())
