# config/ - 설정 파일

## 개요

프로젝트의 모든 JSON 설정 파일을 관리합니다.
카테고리, 브랜드, 임계값, 비즈니스 규칙 등을 코드와 분리하여 유지보수성을 높입니다.

## 파일 목록

| 파일 | 크기 | 설명 |
|------|------|------|
| `thresholds.json` | ~2KB | 카테고리 URL, 알림 임계값 |
| `category_hierarchy.json` | ~1KB | 카테고리 계층 구조 |
| `brands.json` | ~500B | 추적 브랜드 목록 |
| `competitors.json` | ~300B | 주요 경쟁사 |
| `entities.json` | ~1KB | 엔티티 정의 (Product, Brand, etc) |
| `rules.json` | ~800B | 비즈니스 규칙 |
| `asin_brand_mapping.json` | ~5KB | ASIN → 브랜드 매핑 |
| `tracked_competitors.json` | ~200B | 모니터링 대상 경쟁사 |
| `public_apis.json` | ~1KB | 공공 API 설정 |
| `AGENTS.md` | - | 현재 문서 |

## thresholds.json

### 구조

```json
{
  "categories": {
    "beauty": {
      "url": "zgbs/beauty/beauty",
      "node_id": "beauty",
      "level": 0
    },
    "skin_care": {
      "url": "zgbs/beauty/11060451",
      "node_id": "11060451",
      "level": 1,
      "parent_id": "beauty"
    },
    "lip_care": {
      "url": "zgbs/beauty/3761351",
      "node_id": "3761351",
      "level": 2,
      "parent_id": "skin_care"
    },
    "lip_makeup": {
      "url": "zgbs/beauty/11059031",
      "node_id": "11059031",
      "level": 2,
      "parent_id": "makeup"
    },
    "powder": {
      "url": "zgbs/beauty/11058971",
      "node_id": "11058971",
      "level": 3,
      "parent_id": "face_makeup"
    }
  },
  "alerts": {
    "rank_change_threshold": 10,
    "sos_change_threshold": 5.0
  }
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `url` | string | Amazon BSR URL 경로 |
| `node_id` | string | Amazon 카테고리 노드 ID |
| `level` | int | 계층 레벨 (0=L0, 1=L1, ...) |
| `parent_id` | string | 상위 카테고리 ID |

### 사용 예시

```python
import json

with open("config/thresholds.json") as f:
    config = json.load(f)

url = config["categories"]["lip_care"]["url"]
# "zgbs/beauty/3761351"
```

## category_hierarchy.json

카테고리 간 부모-자식 관계를 명시합니다.

```json
{
  "beauty": {
    "name": "Beauty & Personal Care",
    "level": 0,
    "children": ["skin_care", "makeup"]
  },
  "skin_care": {
    "name": "Skin Care",
    "level": 1,
    "parent": "beauty",
    "children": ["lip_care"]
  },
  "lip_care": {
    "name": "Lip Care",
    "level": 2,
    "parent": "skin_care",
    "children": []
  }
}
```

### 계층 시각화

```
beauty (L0)
├── skin_care (L1)
│   └── lip_care (L2)  ← LANEIGE Lip Sleeping Mask
└── makeup (L1)
    ├── lip_makeup (L2)  ← 립스틱, 립글로스
    └── face_makeup (L2)
        └── powder (L3)
```

## brands.json

추적 대상 브랜드 목록입니다.

```json
{
  "target_brand": "LANEIGE",
  "tracked_brands": [
    "LANEIGE",
    "Laneige",
    "Summer Fridays",
    "Aquaphor",
    "Burt's Bees",
    "ChapStick",
    "eos",
    "Vaseline",
    "Lanolips",
    "Blistex"
  ],
  "brand_aliases": {
    "LANEIGE": ["Laneige", "laneige"],
    "Summer Fridays": ["Summer Friday", "SUMMER FRIDAYS"]
  }
}
```

### 사용 예시

```python
brands = json.load(open("config/brands.json"))
target = brands["target_brand"]  # "LANEIGE"
```

## competitors.json

주요 경쟁사 정보입니다.

```json
{
  "direct_competitors": [
    {
      "brand": "Summer Fridays",
      "category": "Premium K-beauty/Lip Care",
      "strength": "Viral TikTok presence"
    }
  ],
  "indirect_competitors": [
    {
      "brand": "Aquaphor",
      "category": "Mass Market Lip Care",
      "strength": "Dermatologist recommended"
    }
  ]
}
```

## entities.json

도메인 엔티티 스키마 정의입니다.

```json
{
  "Product": {
    "required_fields": ["asin", "title", "brand", "rank"],
    "optional_fields": ["price", "rating", "reviews"]
  },
  "Brand": {
    "required_fields": ["name"],
    "optional_fields": ["parent_company", "country"]
  },
  "Category": {
    "required_fields": ["id", "name", "level"],
    "optional_fields": ["parent_id"]
  }
}
```

## rules.json

비즈니스 규칙 정의입니다.

```json
{
  "ranking_rules": {
    "top_tier": { "min_rank": 1, "max_rank": 10 },
    "mid_tier": { "min_rank": 11, "max_rank": 50 },
    "low_tier": { "min_rank": 51, "max_rank": 100 }
  },
  "alert_rules": {
    "critical": "rank_change >= 20",
    "warning": "rank_change >= 10",
    "info": "rank_change >= 5"
  },
  "metric_thresholds": {
    "sos": { "excellent": 50, "good": 30, "fair": 10 },
    "hhi": { "concentrated": 0.25, "moderate": 0.15, "competitive": 0.1 }
  }
}
```

### 사용 예시

```python
rules = json.load(open("config/rules.json"))
if rank <= rules["ranking_rules"]["top_tier"]["max_rank"]:
    tier = "top_tier"
```

## asin_brand_mapping.json

ASIN → 브랜드 매핑 테이블입니다.
크롤링 시 브랜드 정규화에 사용됩니다.

```json
{
  "B0CDJQTY77": "LANEIGE",
  "B0BZ895GS8": "Summer Fridays",
  "B004FHYS8Q": "Aquaphor",
  "B01M4MCUAF": "Burt's Bees"
}
```

### 자동 업데이트

```python
# src/tools/amazon_scraper.py
async def update_asin_mapping(asin: str, brand: str):
    mapping = json.load(open("config/asin_brand_mapping.json"))
    mapping[asin] = brand
    json.dump(mapping, open("config/asin_brand_mapping.json", "w"))
```

## tracked_competitors.json

실시간 모니터링 대상 경쟁사입니다.

```json
{
  "competitors": [
    "Summer Fridays",
    "Aquaphor",
    "Burt's Bees"
  ],
  "monitoring_frequency": "daily"
}
```

## public_apis.json

공공 API 엔드포인트 설정입니다.

```json
{
  "customs": {
    "name": "관세청 수출입무역통계",
    "base_url": "https://unipass.customs.go.kr/openapi",
    "endpoints": {
      "export_by_country": "/v1/export/country",
      "import_by_country": "/v1/import/country"
    },
    "auth": "API_KEY",
    "rate_limit": "1000/day"
  },
  "mfds": {
    "name": "식약처 기능성화장품 API",
    "base_url": "https://apis.data.go.kr/1471000",
    "endpoints": {
      "functional_cosmetics": "/FuncCosmeticInfoService/getFuncCosmeticList"
    },
    "auth": "API_KEY",
    "rate_limit": "unlimited"
  }
}
```

### 사용 예시

```python
# src/tools/public_data_collector.py
config = json.load(open("config/public_apis.json"))
base_url = config["customs"]["base_url"]
endpoint = config["customs"]["endpoints"]["export_by_country"]
url = f"{base_url}{endpoint}"
```

## 설정 로딩 패턴

### 1. 직접 로딩

```python
import json

def load_config(filename: str) -> dict:
    with open(f"config/{filename}") as f:
        return json.load(f)

brands = load_config("brands.json")
```

### 2. 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    with open("config/thresholds.json") as f:
        return json.load(f)
```

### 3. Pydantic 모델

```python
from pydantic import BaseModel

class CategoryConfig(BaseModel):
    url: str
    node_id: str
    level: int
    parent_id: Optional[str] = None

# 검증된 로딩
config = CategoryConfig(**raw_data)
```

## 수정 가이드

### 1. 카테고리 추가

```json
// thresholds.json
{
  "categories": {
    "new_category": {
      "url": "zgbs/beauty/12345678",
      "node_id": "12345678",
      "level": 2,
      "parent_id": "makeup"
    }
  }
}
```

### 2. 브랜드 추가

```json
// brands.json
{
  "tracked_brands": [
    "LANEIGE",
    "New Brand"  // 여기에 추가
  ]
}
```

### 3. 임계값 조정

```json
// thresholds.json
{
  "alerts": {
    "rank_change_threshold": 15,  // 10 → 15로 변경
    "sos_change_threshold": 3.0   // 5.0 → 3.0으로 변경
  }
}
```

## 주의사항

1. **JSON 유효성**: 수정 후 반드시 JSON 문법 검증
2. **백업**: 수정 전 원본 백업
3. **동기화**: Railway 배포 시 config/ 파일도 함께 배포
4. **타입 안전성**: Pydantic 모델로 검증 권장
5. **주석 불가**: JSON은 주석을 지원하지 않음 (JSONC 또는 YAML 고려)

## 검증 스크립트

```python
# scripts/validate_config.py
import json
from pathlib import Path

def validate_all_configs():
    config_dir = Path("config")
    for json_file in config_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                json.load(f)
            print(f"✅ {json_file.name}")
        except json.JSONDecodeError as e:
            print(f"❌ {json_file.name}: {e}")

if __name__ == "__main__":
    validate_all_configs()
```

## 참고

- `src/shared/constants.py` - 상수 참조
- `src/tools/amazon_scraper.py` - 카테고리 URL 사용
- `src/agents/alert_agent.py` - 알림 임계값 사용
- `CLAUDE.md` - 카테고리 계층 구조
