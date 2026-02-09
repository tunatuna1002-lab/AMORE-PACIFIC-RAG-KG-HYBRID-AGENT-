# src/infrastructure - DI & Persistence

## OVERVIEW

Dependency injection container, configuration management, and repository implementations for Google Sheets and local JSON storage.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| Container | `container.py` | DI wiring: KG + reasoners + retrievers |
| Bootstrap | `bootstrap.py` | App initialization + component setup |
| ConfigManager | `config/config_manager.py` | Env vars + JSON config loading |
| SheetsRepository | `persistence/sheets_repository.py` | Google Sheets storage (gspread) |
| JsonRepository | `persistence/json_repository.py` | Local JSON file storage |

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new DI component | `container.py` | Register in container |
| Change config loading | `config/config_manager.py` | `AppConfig.from_env()` |
| Add persistence impl | `persistence/` | Implement domain Repository protocol |

## CONFIGURATION

### Environment Variables (via ConfigManager)
```python
OPENAI_API_KEY          # Required
API_KEY                 # API auth
GOOGLE_SPREADSHEET_ID   # Sheets integration
AUTO_START_SCHEDULER    # Auto-start on boot
PORT                    # Server port
```

### Config Files
- `config/thresholds.json` - Alert thresholds
- `config/brands.json` - Brand metadata

## PERSISTENCE PATTERNS

| Repository | Backend | Worksheets/Files |
|------------|---------|------------------|
| SheetsRepository | Google Sheets | RankRecords, BrandMetrics, MarketMetrics |
| JsonRepository | Local JSON | `data/*.json` |

### Sheets Auth
```python
# Service account credentials via GOOGLE_APPLICATION_CREDENTIALS
# or GOOGLE_SHEETS_CREDENTIALS_JSON (base64)
```

## ANTI-PATTERNS

- **NEVER** import infrastructure in domain layer
- **NEVER** hardcode credentials (use env vars)
- **NEVER** bypass ConfigManager for config access
- **NEVER** instantiate repositories directly (use DI container)
