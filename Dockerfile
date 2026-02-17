# Python 3.11 기반 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (Playwright용 + 한글 폰트)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    fonts-noto-cjk \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright 브라우저 설치
RUN playwright install chromium --with-deps

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8001

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 서버 실행 (start.py 스크립트 사용)
CMD ["python", "scripts/start.py"]
