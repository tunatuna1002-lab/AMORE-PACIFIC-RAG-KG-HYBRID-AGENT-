# Tracked Competitors Integration - Learnings

## Implementation Date
2026-01-23

## Problem Statement
Summer Fridays was showing "-" in the dashboard competitor comparison because:
1. `crawler_agent.py` scrapes tracked competitors and stores in `competitor_products` table
2. `dashboard_exporter.py` only used organic Top 100 crawl results
3. Summer Fridays had 0.7% SoS (only 2 products in Top 100), not making the top 10 cutoff

## Solution Implemented

### 1. Added `_load_tracked_competitors()` Method
**Location**: `src/tools/dashboard_exporter.py` (lines ~950-1010)

**Functionality**:
- Primary: Load from SQLite `competitor_products` table (latest snapshot)
- Fallback: Load from `config/tracked_competitors.json`
- Returns: Brand → {product_count, avg_price, tier}

**Key Logic**:
```python
# Async-safe loading (handles existing event loops)
if loop.is_running():
    # Direct SQLite query
else:
    # Async call to get_competitor_products()
```

### 2. Modified `_generate_competitor_data()` Method
**Location**: `src/tools/dashboard_exporter.py` (lines ~452-495)

**Key Changes**:
1. Load tracked competitors config
2. Mark organic competitors with `is_tracked: True` if they're in tracked list
3. Add tracked competitors not in Top 100 with `sos: 0, avg_rank: None`
4. **Prioritization logic**: Ensure tracked competitors always appear
   - Take top 7 non-tracked + all tracked (up to 10 total)
   - Resort by SoS

### 3. Data Flow

Top 100 Organic Results → Mark Tracked Brands → Sort by SoS → Prioritize Tracked → Final Top 10

## Test Results

**Before**:
- Summer Fridays: NOT in top 10 (0.7% SoS < 2.3% cutoff)

**After**:
- Summer Fridays: Position #10 [TRACKED]
- SoS: 0.7%, Product Count: 2, Avg Price: $24.00, Avg Rank: 39.5
- `is_tracked: true` flag in JSON

## Edge Cases Handled

1. Summer Fridays in Top 100 but low SoS: Prioritized to appear
2. Summer Fridays not in Top 100: Would show sos: 0, avg_rank: null
3. SQLite empty + config exists: Fallback to config works
4. Event loop already running: Sync SQLite query instead of async

## Related Files Modified

1. `src/tools/dashboard_exporter.py` (+80 lines)
   - Added `_load_tracked_competitors()`
   - Modified `_generate_competitor_data()`

## Verification

- [x] Summer Fridays appears in dashboard JSON
- [x] `is_tracked: true` flag set correctly
- [x] SoS, avg_rank, avg_price calculated from organic data
- [x] Fallback to config works when SQLite empty
- [x] No breaking changes to existing competitor logic
