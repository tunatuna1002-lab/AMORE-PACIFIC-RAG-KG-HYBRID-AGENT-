# Category Expansion Guide

## Overview

This document outlines the available Amazon Best Sellers sub-categories that can be added to the tracking system, focusing on categories relevant to LANEIGE and AMOREPACIFIC product lines.

---

## Current Tracking Categories (5)

| Category ID | Name | URL Path | Products |
|-------------|------|----------|----------|
| `beauty` | Beauty & Personal Care | /zgbs/beauty | 100 |
| `skin_care` | Skin Care | /zgbs/beauty/11060451 | 100 |
| `lip_care` | Lip Care | /zgbs/beauty/3761351 | 100 |
| `lip_makeup` | Lip Makeup | /zgbs/beauty/11059031 | 100 |
| `face_powder` | Face Powder | /zgbs/beauty/11058971 | 100 |

**Total SKUs Currently Tracked: 500**

---

## Recommended Expansion Categories

### Priority 1: LANEIGE Core Categories

| Category ID | Name | URL Path | Relevance |
|-------------|------|----------|-----------|
| `facial_moisturizers` | Face Moisturizers | /zgbs/beauty/11060711 | Water Bank, Cica products |
| `facial_serums` | Face Serums | /zgbs/beauty/11060721 | Water Bank Serum |
| `sleeping_masks` | Sleeping Masks | /zgbs/beauty/7792263011 | Lip Sleeping Mask, Water Sleeping Mask |
| `toners` | Toners | /zgbs/beauty/11060731 | Cream Skin Toner |
| `facial_cleansers` | Facial Cleansers | /zgbs/beauty/11060741 | Cream Skin Cleanser |

### Priority 2: Adjacent Categories

| Category ID | Name | URL Path | Relevance |
|-------------|------|----------|-----------|
| `eye_creams` | Eye Creams | /zgbs/beauty/11060761 | Eye sleeping mask |
| `sunscreens` | Sunscreens | /zgbs/beauty/11060781 | UV protection products |
| `face_primers` | Face Primers | /zgbs/beauty/11058961 | Makeup base products |
| `bb_cc_creams` | BB & CC Creams | /zgbs/beauty/11058951 | Cushion products |
| `makeup_setting_sprays` | Setting Sprays | /zgbs/beauty/11059001 | Finish products |

### Priority 3: K-Beauty Focus

| Category ID | Name | URL Path | Relevance |
|-------------|------|----------|-----------|
| `k_beauty_skin` | K-Beauty Skin Care | /zgbs/beauty/7730906011 | All K-beauty competitors |
| `sheet_masks` | Sheet Masks | /zgbs/beauty/7792264011 | Sheet mask products |
| `essences` | Essences | /zgbs/beauty/7792261011 | Korean essence category |

---

## Implementation Guide

### Adding a New Category

1. Update `config/thresholds.json`:

```json
"categories": {
    // ... existing categories ...
    "facial_moisturizers": {
        "url": "https://www.amazon.com/Best-Sellers-Beauty-Facial-Moisturizers/zgbs/beauty/11060711",
        "name": "Facial Moisturizers"
    }
}
```

2. Run crawling to collect data:
```bash
python -m src.tools.amazon_scraper
```

3. Verify data in Google Sheets and dashboard.

---

## Capacity Considerations

| Configuration | Categories | SKUs/Day | API Calls | Est. Time |
|--------------|------------|----------|-----------|-----------|
| Current | 5 | 500 | ~10 | ~5 min |
| Recommended | 10 | 1,000 | ~20 | ~10 min |
| Maximum | 15 | 1,500 | ~30 | ~15 min |

**Notes:**
- Each category requires 2 page loads (1-50, 51-100)
- Minimum 2 seconds delay between requests (rate limiting)
- Daily crawl at 06:00 KST recommended

---

## Configuration Example

For LANEIGE-focused tracking (recommended 10 categories):

```json
{
    "categories": {
        "beauty": { "url": "...", "name": "Beauty & Personal Care" },
        "skin_care": { "url": "...", "name": "Skin Care" },
        "lip_care": { "url": "...", "name": "Lip Care" },
        "lip_makeup": { "url": "...", "name": "Lip Makeup" },
        "face_powder": { "url": "...", "name": "Face Powder" },
        "facial_moisturizers": { "url": "...", "name": "Facial Moisturizers" },
        "facial_serums": { "url": "...", "name": "Facial Serums" },
        "sleeping_masks": { "url": "...", "name": "Sleeping Masks" },
        "toners": { "url": "...", "name": "Toners" },
        "k_beauty_skin": { "url": "...", "name": "K-Beauty Skin Care" }
    }
}
```

---

## Dashboard Updates Required

When adding new categories:

1. Update category tabs in `dashboard/amore_unified_dashboard_v4.html`
2. Add category to `selectCategory()` function
3. Update KPI calculations in `src/ontology/unified_brain.py`
4. Verify Google Sheets schema supports new category

---

## Performance Impact

| Metric | 5 Categories | 10 Categories | 15 Categories |
|--------|--------------|---------------|---------------|
| Crawl Time | ~5 min | ~10 min | ~15 min |
| Data Size | ~500 KB/day | ~1 MB/day | ~1.5 MB/day |
| API Response | <500ms | <800ms | <1200ms |
| Chart Load | Instant | Instant | ~1s |

---

## Recommendation

**For AMOREPACIFIC competition demo:**
- Start with current 5 categories (500 SKUs)
- Expand to 10 categories if time permits
- Focus on categories where LANEIGE has presence

**Post-competition:**
- Implement full 15 category tracking
- Add scheduled category refresh
- Consider regional expansion (Amazon Japan, Europe)

---

Last updated: 2026-01-18
