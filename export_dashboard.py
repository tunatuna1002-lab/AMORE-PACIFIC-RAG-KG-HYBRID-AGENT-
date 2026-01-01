"""
Dashboard Data Export Script
Google Sheets 데이터를 Dashboard용 JSON으로 내보내기

Usage:
    python export_dashboard.py
"""

import asyncio
import os
from dotenv import load_dotenv

from src.tools.dashboard_exporter import DashboardExporter

# 환경 변수 로드
load_dotenv()


async def main():
    """대시보드 데이터 내보내기 실행"""
    print("=" * 50)
    print("Dashboard Data Export")
    print("=" * 50)

    # Exporter 초기화
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
    exporter = DashboardExporter(spreadsheet_id=spreadsheet_id)

    print(f"\nSpreadsheet ID: {spreadsheet_id}")
    print("Initializing...")

    success = await exporter.initialize()
    if not success:
        print("Failed to initialize exporter!")
        return

    print("Exporting dashboard data...")

    # 데이터 내보내기
    output_path = "./data/dashboard_data.json"
    result = await exporter.export_dashboard_data(output_path)

    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    # 결과 출력
    print("\n" + "=" * 50)
    print("Export Complete!")
    print("=" * 50)
    print(f"\nOutput: {output_path}")
    print(f"\nMetadata:")
    metadata = result.get("metadata", {})
    print(f"  - Generated at: {metadata.get('generated_at', 'N/A')}")
    print(f"  - Data date: {metadata.get('data_date', 'N/A')}")
    print(f"  - Total products: {metadata.get('total_products', 0)}")
    print(f"  - LANEIGE products: {metadata.get('laneige_products', 0)}")

    print(f"\nHome Page:")
    home = result.get("home", {})
    print(f"  - Status: {home.get('status', {}).get('exposure', 'N/A')}")
    print(f"  - Action items: {len(home.get('action_items', []))}")

    print(f"\nBrand KPIs:")
    kpis = result.get("brand", {}).get("kpis", {})
    print(f"  - SoS: {kpis.get('sos', 0)}%")
    print(f"  - Top10 Count: {kpis.get('top10_count', 0)}")
    print(f"  - Avg Rank: {kpis.get('avg_rank', 0)}")
    print(f"  - HHI: {kpis.get('hhi', 0)}")

    print(f"\nCategories:")
    for cat_id, cat_data in result.get("categories", {}).items():
        print(f"  - {cat_data.get('name', cat_id)}: SoS {cat_data.get('sos', 0)}%, Best Rank #{cat_data.get('best_rank', 'N/A')}")

    print(f"\nProducts:")
    for asin, product in list(result.get("products", {}).items())[:5]:
        print(f"  - {product.get('name', 'Unknown')[:30]}: Rank #{product.get('rank', 'N/A')}")

    print("\n" + "=" * 50)
    print("Dashboard can now load data from:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
