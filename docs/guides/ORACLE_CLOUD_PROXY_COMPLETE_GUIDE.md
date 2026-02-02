# 🚀 Oracle Cloud 무료 프록시 서버 완전 가이드

> **목표**: Oracle Cloud Free Tier를 이용해 무료 프록시 서버를 구축하고, Amazon 크롤링에 활용
>
> **난이도**: ⭐⭐ (초보자도 가능)
>
> **예상 소요 시간**: 1~2시간
>
> **비용**: $0 (영구 무료)

---

## 📋 목차

1. [사전 준비물](#1-사전-준비물)
2. [Oracle Cloud 회원가입](#2-oracle-cloud-회원가입-상세)
3. [VM 인스턴스 생성](#3-vm-인스턴스-생성)
4. [네트워크 보안 설정](#4-네트워크-보안-설정)
5. [Squid 프록시 설치](#5-squid-프록시-설치)
6. [Python 코드 연동](#6-python-코드-연동)
7. [테스트 및 검증](#7-테스트-및-검증)
8. [Oracle vs GCP 비교](#8-oracle-cloud-vs-google-cloud-비교)
9. [Git 브랜치 관리](#9-git-브랜치-관리-가이드)
10. [문제 해결](#10-문제-해결-troubleshooting)

---

## 1. 사전 준비물

### 필수 항목

| 항목 | 설명 | 비고 |
|------|------|------|
| 이메일 | Gmail, 네이버 등 | 인증용 |
| 휴대폰 번호 | SMS 인증 | 한국 번호 가능 |
| 신용카드/체크카드 | 본인 명의 | 결제 안 됨, 인증용 |
| SSH 클라이언트 | 터미널 접속용 | Mac: 기본 터미널 / Windows: PowerShell |

### 신용카드 관련 중요 사항

```
⚠️ 걱정 마세요!
- 카드 등록은 "본인 확인"용입니다
- Always Free 계정은 절대 과금되지 않습니다
- "Upgrade" 버튼을 누르지 않는 한 무료입니다
- 1달러 임시 결제 후 즉시 취소됩니다 (한도 확인용)
```

---

## 2. Oracle Cloud 회원가입 (상세)

### Step 2.1: Oracle Cloud 웹사이트 접속

1. 브라우저에서 접속: **https://www.oracle.com/cloud/free/**

2. **"Start for free"** 버튼 클릭

![Oracle Free Tier 메인](https://oracle-cloud-screenshot-placeholder.com/main)

---

### Step 2.2: 계정 정보 입력

#### 기본 정보 입력 화면

```
┌─────────────────────────────────────────────────────────┐
│  Create Your Free Account                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Country/Territory: [South Korea          ▼]            │
│                                                          │
│  First Name:        [동원                    ]          │
│  Last Name:         [이                      ]          │
│                                                          │
│  Email:             [your-email@gmail.com    ]          │
│                                                          │
│  ☑ I have read and agree to the terms                   │
│                                                          │
│  [        Verify my email        ]                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**입력값:**
- Country: `South Korea`
- First Name: 영문 이름 (예: `Dongwon`)
- Last Name: 영문 성 (예: `Lee`)
- Email: 실제 사용하는 이메일

3. **"Verify my email"** 클릭

4. 이메일 확인 → 인증 링크 클릭

---

### Step 2.3: 비밀번호 및 상세 정보

```
┌─────────────────────────────────────────────────────────┐
│  Account Details                                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Cloud Account Name: [amore-proxy-server    ]           │
│  (영문, 숫자, 하이픈만 가능)                               │
│                                                          │
│  Home Region:        [South Korea Central (Seoul) ▼]    │
│  ⚠️ 중요: 나중에 변경 불가! 가까운 지역 선택               │
│                                                          │
│  Password:           [••••••••••••••        ]           │
│  (대문자+소문자+숫자+특수문자 조합)                        │
│                                                          │
│  Confirm Password:   [••••••••••••••        ]           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Home Region 선택 (중요!):**

| 리전 | 위치 | 추천 |
|------|------|------|
| `South Korea Central (Seoul)` | 서울 | ⭐ 최추천 (가장 빠름) |
| `Japan East (Tokyo)` | 도쿄 | 차선책 |
| `US West (Phoenix)` | 미국 서부 | Amazon US 크롤링 시 |

> 💡 **팁**: 한국에서 사용한다면 `Seoul` 선택. Amazon US 크롤링이 주 목적이라면 `US` 리전도 고려.

**비밀번호 규칙:**
- 최소 8자
- 대문자 1개 이상
- 소문자 1개 이상
- 숫자 1개 이상
- 특수문자 1개 이상
- 예: `MyProxy2024!`

---

### Step 2.4: 주소 및 연락처

```
┌─────────────────────────────────────────────────────────┐
│  Address Information                                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Address Line 1:    [123 Gangnam-daero      ]           │
│  City:              [Seoul                  ]           │
│  State/Province:    [Seoul                  ]           │
│  Postal Code:       [06100                  ]           │
│  Phone Number:      [+82 10-1234-5678       ]           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**입력 예시:**
- Address: 영문 주소 (대략적으로 OK)
- City: `Seoul`
- Postal Code: 우편번호 5자리
- Phone: `+82 10-XXXX-XXXX` 형식

---

### Step 2.5: 신용카드 등록 (결제 안 됨!)

```
┌─────────────────────────────────────────────────────────┐
│  Payment Verification                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ⚠️ This is for verification only.                      │
│     You will NOT be charged.                            │
│                                                          │
│  Card Number:       [1234-5678-9012-3456    ]           │
│  Expiration:        [MM/YY]                             │
│  CVV:               [123 ]                              │
│                                                          │
│  ☑ Add my Free Cloud credits ($300 for 30 days)        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**카드 등록 시 알아야 할 것:**

| 질문 | 답변 |
|------|------|
| 실제로 결제되나요? | ❌ 아니요, $1 임시 승인 후 즉시 취소 |
| 체크카드도 되나요? | ✅ 네, 해외결제 가능한 카드면 OK |
| 나중에 과금될 수 있나요? | ❌ Always Free 계정은 절대 과금 안 됨 |
| Upgrade하면요? | ⚠️ 그때만 과금 시작 (누르지 마세요) |

---

### Step 2.6: 계정 생성 완료

```
┌─────────────────────────────────────────────────────────┐
│  ✅ Your account is being created                       │
│                                                          │
│  Please wait 5-10 minutes...                            │
│                                                          │
│  You will receive an email when ready.                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

- 계정 생성까지 **5~10분** 소요
- 이메일로 "Your Oracle Cloud account is ready" 수신
- **"Sign in to Oracle Cloud"** 클릭

---

## 3. VM 인스턴스 생성

### Step 3.1: Oracle Cloud Console 접속

1. https://cloud.oracle.com 접속
2. Cloud Account Name 입력 (예: `amore-proxy-server`)
3. 이메일/비밀번호로 로그인

---

### Step 3.2: Compute Instance 생성 시작

1. 좌측 상단 **☰ (햄버거 메뉴)** 클릭
2. **Compute** → **Instances** 클릭
3. **"Create instance"** 버튼 클릭

---

### Step 3.3: 인스턴스 기본 설정

```
┌─────────────────────────────────────────────────────────┐
│  Create compute instance                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Name: [proxy-server-1                     ]            │
│                                                          │
│  Compartment: [root (기본값)               ▼]           │
│                                                          │
│  Placement                                               │
│  ├─ Availability domain: AD-1 (기본값 OK)               │
│  └─ Fault domain: (자동)                                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**이름 규칙:**
- 첫 번째 서버: `proxy-server-1`
- 두 번째 서버: `proxy-server-2`

---

### Step 3.4: 이미지 및 Shape 선택 (⚠️ 중요!)

#### Image 선택

1. **"Edit"** 클릭 (Image and shape 섹션)
2. **"Change image"** 클릭
3. 다음 중 선택:

| 이미지 | 추천 | 이유 |
|--------|------|------|
| **Ubuntu 22.04** | ⭐ 최추천 | 자료 많음, Squid 설치 쉬움 |
| Oracle Linux 8 | 추천 | Oracle 최적화 |
| CentOS 8 | 가능 | 익숙하다면 |

> **선택**: `Canonical Ubuntu 22.04` → **"Select image"** 클릭

#### Shape 선택 (무료 범위 확인!)

1. **"Change shape"** 클릭
2. **"Specialty and previous generation"** 탭 선택
3. **`VM.Standard.E2.1.Micro`** 선택 ← ⚠️ **이것만 무료!**

```
┌─────────────────────────────────────────────────────────┐
│  Shape Selection                                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ☐ AMD Rome  (Flexible) - 유료                          │
│  ☐ Intel Ice Lake (Flexible) - 유료                     │
│  ☑ VM.Standard.E2.1.Micro ← Always Free 🆓             │
│     • 1 OCPU                                            │
│     • 1 GB Memory                                       │
│     • 0.48 Gbps network bandwidth                       │
│                                                          │
│  또는                                                    │
│                                                          │
│  ☑ VM.Standard.A1.Flex (Ampere ARM) ← Always Free 🆓   │
│     • 최대 4 OCPU (무료 범위 내)                         │
│     • 최대 24 GB Memory (무료 범위 내)                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Shape 비교:**

| Shape | CPU | 메모리 | 무료 개수 | 추천 |
|-------|-----|--------|----------|------|
| `VM.Standard.E2.1.Micro` | 1 OCPU (AMD) | 1 GB | 2개 | ⭐ 안정적 |
| `VM.Standard.A1.Flex` | 4 OCPU (ARM) | 24 GB | 1개 | 고성능 필요시 |

> **선택**: `VM.Standard.E2.1.Micro` → **"Select shape"** 클릭

---

### Step 3.5: 네트워킹 설정

```
┌─────────────────────────────────────────────────────────┐
│  Networking                                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Virtual cloud network: [Create new VCN    ▼]           │
│    Name: [proxy-vcn                       ]             │
│                                                          │
│  Subnet: [Create new public subnet        ▼]           │
│    Name: [proxy-subnet                    ]             │
│                                                          │
│  ☑ Assign a public IPv4 address ← 반드시 체크!          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

> ⚠️ **중요**: "Assign a public IPv4 address" 반드시 체크!
> 이게 없으면 외부에서 접속 불가

---

### Step 3.6: SSH 키 설정

```
┌─────────────────────────────────────────────────────────┐
│  Add SSH keys                                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ○ Generate a key pair for me ← 가장 쉬움               │
│  ○ Upload public key files (.pub)                       │
│  ○ Paste public keys                                    │
│                                                          │
│  [        Save Private Key        ] ← 반드시 다운로드!  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**SSH 키 옵션:**

| 옵션 | 설명 | 추천 |
|------|------|------|
| Generate key pair | Oracle이 자동 생성 | ⭐ 초보자 추천 |
| Upload public key | 기존 키 업로드 | 기존 키 있으면 |
| Paste public keys | 키 내용 붙여넣기 | 고급 사용자 |

**"Generate a key pair for me" 선택 시:**

1. **"Save Private Key"** 클릭 → `ssh-key-*.key` 다운로드
2. 이 파일 **절대 잃어버리지 마세요!** (서버 접속 유일한 방법)
3. 안전한 곳에 보관 (예: `~/.ssh/oracle-proxy.key`)

---

### Step 3.7: 부팅 볼륨 설정 (기본값 OK)

```
┌─────────────────────────────────────────────────────────┐
│  Boot volume                                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Boot volume size: [50 GB] (기본값, 무료 범위 내)       │
│                                                          │
│  ☐ Specify a custom boot volume size                    │
│  ☐ Use in-transit encryption                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

> 기본값 그대로 두면 됩니다 (50GB, 무료 범위)

---

### Step 3.8: 인스턴스 생성

1. **"Create"** 버튼 클릭
2. 상태가 `PROVISIONING` → `RUNNING` 으로 변경 (2~3분)
3. **Public IP 주소 확인** (예: `129.154.xxx.xxx`)

```
┌─────────────────────────────────────────────────────────┐
│  Instance Details                                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Name:          proxy-server-1                          │
│  State:         🟢 RUNNING                              │
│                                                          │
│  Primary VNIC                                            │
│  ├─ Public IP:  129.154.xxx.xxx  ← 📋 복사해두세요!     │
│  └─ Private IP: 10.0.0.xxx                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 네트워크 보안 설정

### Step 4.1: Security List 접근

1. **☰ 메뉴** → **Networking** → **Virtual Cloud Networks**
2. 생성한 VCN 클릭 (예: `proxy-vcn`)
3. **Resources** → **Security Lists** 클릭
4. **Default Security List** 클릭

---

### Step 4.2: Ingress Rule 추가 (외부 → 서버)

1. **"Add Ingress Rules"** 클릭

```
┌─────────────────────────────────────────────────────────┐
│  Add Ingress Rules                                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Stateless:        ☐ (체크 해제)                        │
│                                                          │
│  Source Type:      [CIDR       ▼]                       │
│  Source CIDR:      [0.0.0.0/0   ] ← 모든 IP 허용        │
│                                                          │
│  IP Protocol:      [TCP        ▼]                       │
│                                                          │
│  Source Port Range: [All       ]                        │
│  Destination Port Range: [3128  ] ← Squid 포트          │
│                                                          │
│  Description:      [Squid Proxy Port]                   │
│                                                          │
│  [        Add Ingress Rules        ]                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**입력값:**
- Source CIDR: `0.0.0.0/0`
- IP Protocol: `TCP`
- Destination Port Range: `3128`
- Description: `Squid Proxy Port`

2. **"Add Ingress Rules"** 클릭

---

### Step 4.3: (선택) 본인 IP만 허용 (보안 강화)

보안을 위해 본인 IP만 허용하려면:

1. https://whatismyip.com 에서 본인 IP 확인
2. Source CIDR에 `YOUR_IP/32` 입력
   - 예: `123.456.789.123/32`

---

## 5. Squid 프록시 설치

### Step 5.1: SSH 접속

#### Mac/Linux 터미널

```bash
# SSH 키 권한 설정 (최초 1회)
chmod 400 ~/Downloads/ssh-key-*.key

# SSH 접속
ssh -i ~/Downloads/ssh-key-*.key ubuntu@129.154.xxx.xxx
```

#### Windows PowerShell

```powershell
# SSH 접속
ssh -i C:\Users\YourName\Downloads\ssh-key-*.key ubuntu@129.154.xxx.xxx
```

**접속 성공 시:**
```
Welcome to Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-1052-oracle x86_64)
...
ubuntu@proxy-server-1:~$
```

---

### Step 5.2: 시스템 업데이트

```bash
# 패키지 목록 업데이트
sudo apt update

# 시스템 업그레이드
sudo apt upgrade -y
```

---

### Step 5.3: Squid 설치

```bash
# Squid 및 관련 패키지 설치
sudo apt install squid apache2-utils -y

# 설치 확인
squid -v
```

**출력 예시:**
```
Squid Cache: Version 5.7
```

---

### Step 5.4: Squid 설정 백업

```bash
# 원본 설정 백업
sudo cp /etc/squid/squid.conf /etc/squid/squid.conf.backup
```

---

### Step 5.5: 프록시 인증 설정

```bash
# 비밀번호 파일 생성
sudo touch /etc/squid/passwd

# 사용자 추가 (proxyuser는 원하는 이름으로 변경 가능)
sudo htpasswd -c /etc/squid/passwd proxyuser
```

**비밀번호 입력:**
```
New password: [비밀번호 입력 - 화면에 안 보임]
Re-type new password: [다시 입력]
Adding password for user proxyuser
```

> 💡 **팁**: 비밀번호는 복잡하게! 예: `Proxy@2024!Secure`

---

### Step 5.6: Squid 설정 파일 작성

```bash
# 설정 파일 편집
sudo nano /etc/squid/squid.conf
```

**기존 내용 모두 삭제 후** 아래 내용 붙여넣기:

```conf
#
# Oracle Cloud Proxy Server Configuration
# For AMORE RAG Project - Amazon Crawling
#

# ========================================
# 인증 설정
# ========================================
auth_param basic program /usr/lib/squid/basic_ncsa_auth /etc/squid/passwd
auth_param basic realm AMORE Proxy Server
auth_param basic credentialsttl 2 hours
acl authenticated proxy_auth REQUIRED

# ========================================
# 포트 설정
# ========================================
http_port 3128

# ========================================
# 접근 제어
# ========================================
# 인증된 사용자만 허용
http_access allow authenticated

# 그 외 모두 거부
http_access deny all

# ========================================
# 익명성 설정 (IP 숨기기)
# ========================================
# 실제 IP 숨기기
forwarded_for off

# 프록시 관련 헤더 제거
request_header_access Via deny all
request_header_access X-Forwarded-For deny all
request_header_access Cache-Control deny all
request_header_access Proxy-Connection deny all

# ========================================
# 캐시 설정 (비활성화 - 실시간 데이터 필요)
# ========================================
cache deny all

# ========================================
# 로그 설정 (최소화)
# ========================================
access_log /var/log/squid/access.log
cache_log /var/log/squid/cache.log

# ========================================
# 성능 설정
# ========================================
# 최대 연결 수
max_filedescriptors 65535

# 연결 타임아웃
connect_timeout 60 seconds
read_timeout 60 seconds
request_timeout 60 seconds

# ========================================
# DNS 설정
# ========================================
dns_nameservers 8.8.8.8 8.8.4.4
```

**저장 및 종료:**
- `Ctrl + O` → Enter (저장)
- `Ctrl + X` (종료)

---

### Step 5.7: Ubuntu 방화벽 설정

```bash
# UFW 방화벽 설치 (없으면)
sudo apt install ufw -y

# SSH 허용 (이거 안 하면 접속 끊김!)
sudo ufw allow 22/tcp

# Squid 포트 허용
sudo ufw allow 3128/tcp

# 방화벽 활성화
sudo ufw enable
```

**확인:**
```bash
sudo ufw status
```

**출력:**
```
Status: active

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW       Anywhere
3128/tcp                   ALLOW       Anywhere
```

---

### Step 5.8: iptables 설정 (Oracle Linux 전용)

Oracle Linux의 경우 iptables도 설정:

```bash
# iptables 규칙 추가
sudo iptables -I INPUT -p tcp --dport 3128 -j ACCEPT

# 규칙 저장
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

---

### Step 5.9: Squid 서비스 시작

```bash
# 설정 검증
sudo squid -k parse

# 문제 없으면 재시작
sudo systemctl restart squid

# 부팅 시 자동 시작
sudo systemctl enable squid

# 상태 확인
sudo systemctl status squid
```

**정상 출력:**
```
● squid.service - Squid Web Proxy Server
     Loaded: loaded (/lib/systemd/system/squid.service; enabled)
     Active: active (running) since Mon 2024-01-29 10:00:00 UTC
```

---

### Step 5.10: 프록시 테스트 (서버 내부)

```bash
# 서버 내부에서 테스트
curl -x http://proxyuser:YOUR_CREDENTIAL@localhost:3128 https://httpbin.org/ip
```

**성공 시:**
```json
{
  "origin": "129.154.xxx.xxx"
}
```

---

## 6. Python 코드 연동

### Step 6.1: 프록시 설정 파일 생성

프로젝트 루트에 `config/proxy_config.json` 생성:

```json
{
  "proxy_pool": [
    {
      "name": "oracle-seoul-1",
      "server": "http://129.154.xxx.xxx:3128",
      "username": "proxyuser",
      "credential": "YOUR_CREDENTIAL_HERE",
      "region": "seoul",
      "enabled": true
    },
    {
      "name": "oracle-seoul-2",
      "server": "http://129.154.yyy.yyy:3128",
      "username": "proxyuser",
      "credential": "YOUR_CREDENTIAL_HERE",
      "region": "seoul",
      "enabled": true
    }
  ],
  "rotation_strategy": "random",
  "retry_on_failure": true,
  "max_retries": 3
}
```

> ⚠️ **보안**: 이 파일을 `.gitignore`에 추가하세요!

---

### Step 6.2: .gitignore 업데이트

```bash
echo "config/proxy_config.json" >> .gitignore
```

---

### Step 6.3: 프록시 매니저 클래스

`src/tools/proxy_manager.py` 생성:

```python
"""
Oracle Cloud 프록시 매니저
무료 프록시 풀을 관리하고 로테이션합니다.
"""

import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """프록시 서버 설정"""
    name: str
    server: str
    username: str
    password: str
    region: str
    enabled: bool = True

    @property
    def playwright_config(self) -> Dict:
        """Playwright용 프록시 설정 반환"""
        return {
            "server": self.server,
            "username": self.username,
            "password": self.password
        }

    @property
    def requests_config(self) -> Dict:
        """requests 라이브러리용 프록시 설정 반환"""
        auth = f"{self.username}:{self.password}"
        server = self.server.replace("http://", "")
        return {
            "http": f"http://{auth}@{server}",
            "https": f"http://{auth}@{server}"
        }


class ProxyManager:
    """
    Oracle Cloud 프록시 풀 매니저

    Features:
    - 프록시 로테이션 (random, round-robin)
    - 실패한 프록시 자동 비활성화
    - 헬스 체크
    """

    def __init__(self, config_path: str = "config/proxy_config.json"):
        self.config_path = Path(config_path)
        self.proxies: List[ProxyConfig] = []
        self.current_index = 0
        self.failed_proxies: Dict[str, int] = {}  # name -> fail_count
        self._load_config()

    def _load_config(self):
        """설정 파일 로드"""
        if not self.config_path.exists():
            logger.warning(f"프록시 설정 파일 없음: {self.config_path}")
            return

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        self.proxies = [
            ProxyConfig(**proxy)
            for proxy in config.get("proxy_pool", [])
            if proxy.get("enabled", True)
        ]

        self.rotation_strategy = config.get("rotation_strategy", "random")
        self.max_retries = config.get("max_retries", 3)

        logger.info(f"프록시 {len(self.proxies)}개 로드됨")

    def get_proxy(self) -> Optional[ProxyConfig]:
        """
        사용 가능한 프록시 반환

        Returns:
            ProxyConfig or None
        """
        active_proxies = [p for p in self.proxies if p.enabled]

        if not active_proxies:
            logger.warning("사용 가능한 프록시 없음")
            return None

        if self.rotation_strategy == "random":
            return random.choice(active_proxies)
        else:  # round-robin
            proxy = active_proxies[self.current_index % len(active_proxies)]
            self.current_index += 1
            return proxy

    def report_failure(self, proxy_name: str):
        """
        프록시 실패 보고
        max_retries 초과 시 비활성화
        """
        self.failed_proxies[proxy_name] = self.failed_proxies.get(proxy_name, 0) + 1

        if self.failed_proxies[proxy_name] >= self.max_retries:
            for proxy in self.proxies:
                if proxy.name == proxy_name:
                    proxy.enabled = False
                    logger.warning(f"프록시 비활성화: {proxy_name}")
                    break

    def report_success(self, proxy_name: str):
        """프록시 성공 보고 - 실패 카운트 리셋"""
        self.failed_proxies[proxy_name] = 0

    def get_stats(self) -> Dict:
        """프록시 풀 통계"""
        return {
            "total": len(self.proxies),
            "active": len([p for p in self.proxies if p.enabled]),
            "failed_counts": self.failed_proxies
        }


# 싱글톤 인스턴스
_proxy_manager: Optional[ProxyManager] = None


def get_proxy_manager() -> ProxyManager:
    """전역 프록시 매니저 인스턴스 반환"""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager
```

---

### Step 6.4: amazon_scraper.py 수정

기존 `src/tools/amazon_scraper.py`에 프록시 지원 추가:

```python
# 상단에 import 추가
from src.tools.proxy_manager import get_proxy_manager, ProxyConfig

class AmazonBestsellerScraper:
    def __init__(self, use_proxy: bool = True):
        # ... 기존 코드 ...
        self.use_proxy = use_proxy
        self.proxy_manager = get_proxy_manager() if use_proxy else None

    async def _create_browser_context(self):
        """프록시를 사용하는 브라우저 컨텍스트 생성"""

        launch_options = {
            "headless": True,
        }

        context_options = {
            "user_agent": self._get_random_user_agent(),
            "viewport": {"width": 1920, "height": 1080},
        }

        # 프록시 설정 추가
        if self.use_proxy and self.proxy_manager:
            proxy = self.proxy_manager.get_proxy()
            if proxy:
                launch_options["proxy"] = proxy.playwright_config
                logger.info(f"프록시 사용: {proxy.name}")

        browser = await self.playwright.chromium.launch(**launch_options)
        context = await browser.new_context(**context_options)

        return browser, context

    async def scrape_with_retry(self, category_id: str, max_retries: int = 3):
        """프록시 로테이션으로 재시도하는 스크래핑"""

        for attempt in range(max_retries):
            proxy = self.proxy_manager.get_proxy() if self.use_proxy else None

            try:
                result = await self._scrape_category(category_id, proxy)

                if proxy:
                    self.proxy_manager.report_success(proxy.name)

                return result

            except Exception as e:
                logger.warning(f"스크래핑 실패 (시도 {attempt + 1}): {e}")

                if proxy:
                    self.proxy_manager.report_failure(proxy.name)

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프

        raise Exception(f"모든 재시도 실패: {category_id}")
```

---

## 7. 테스트 및 검증

### Step 7.1: 로컬에서 프록시 테스트

```bash
# 터미널에서 직접 테스트
curl -x http://proxyuser:YOUR_CREDENTIAL@129.154.xxx.xxx:3128 \
     https://httpbin.org/ip
```

**예상 결과:**
```json
{
  "origin": "129.154.xxx.xxx"
}
```

---

### Step 7.2: Python 테스트 스크립트

`tests/test_proxy_connection.py` 생성:

```python
"""프록시 연결 테스트"""

import asyncio
import pytest
from playwright.async_api import async_playwright

# 프록시 설정 (테스트용)
PROXY_CONFIG = {
    "server": "http://129.154.xxx.xxx:3128",
    "username": "proxyuser",
    "credential": "YOUR_CREDENTIAL"
}


@pytest.mark.asyncio
async def test_proxy_connection():
    """프록시를 통한 기본 연결 테스트"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy=PROXY_CONFIG
        )

        page = await browser.new_page()

        # IP 확인 사이트 접속
        await page.goto("https://httpbin.org/ip")
        content = await page.content()

        # Oracle Cloud IP가 표시되는지 확인
        assert "129.154" in content, "프록시 IP가 아님!"

        await browser.close()
        print("✅ 프록시 연결 테스트 성공!")


@pytest.mark.asyncio
async def test_amazon_access():
    """프록시를 통한 Amazon 접속 테스트"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy=PROXY_CONFIG
        )

        page = await browser.new_page()

        # Amazon 베스트셀러 페이지 접속
        await page.goto(
            "https://www.amazon.com/Best-Sellers-Beauty/zgbs/beauty",
            timeout=30000
        )

        # 페이지 로드 확인
        title = await page.title()
        assert "Best Sellers" in title or "Amazon" in title

        await browser.close()
        print("✅ Amazon 접속 테스트 성공!")


if __name__ == "__main__":
    asyncio.run(test_proxy_connection())
    asyncio.run(test_amazon_access())
```

**테스트 실행:**
```bash
python -m pytest tests/test_proxy_connection.py -v
```

---

### Step 7.3: 프록시 상태 모니터링

```bash
# SSH로 서버 접속 후
# 실시간 로그 확인
sudo tail -f /var/log/squid/access.log

# 접속 통계
sudo squidclient -h localhost mgr:info
```

---

## 8. Oracle Cloud vs Google Cloud 비교

### 8.1 비용 비교

| 항목 | Oracle Cloud | Google Cloud |
|------|-------------|--------------|
| **무료 VM** | 2개 (영구) | 1개 (e2-micro) |
| **외부 IP** | 무료 | $3.60/월 ❌ |
| **네트워크 (Egress)** | 10TB 무료 | 1GB 무료 ❌ |
| **스토리지** | 200GB 무료 | 30GB 무료 |
| **총 월 비용** | **$0** | ~$5-10 |

### 8.2 성능 비교

| 항목 | Oracle Cloud | Google Cloud |
|------|-------------|--------------|
| **CPU** | 1 OCPU (AMD EPYC) | 0.25 vCPU (공유) |
| **메모리** | 1GB | 1GB |
| **네트워크** | 480 Mbps | 제한적 |
| **리전 (한국)** | ✅ 서울 | ❌ 없음 (도쿄) |

### 8.3 설치 편의성 비교

| 항목 | Oracle Cloud | Google Cloud |
|------|-------------|--------------|
| **회원가입** | 보통 (5분) | 쉬움 (3분) |
| **VM 생성** | 보통 (10분) | 쉬움 (5분) |
| **방화벽 설정** | 복잡 (2단계) | 쉬움 (1단계) |
| **SSH 접속** | 키 다운로드 필요 | Cloud Shell 제공 |
| **문서/자료** | 적음 | 많음 |
| **전체 난이도** | ⭐⭐⭐ | ⭐⭐ |

### 8.4 최종 추천

| 상황 | 추천 |
|------|------|
| **비용 $0 필수** | ✅ Oracle Cloud |
| **빠른 설정 필요** | ✅ Google Cloud |
| **한국 리전 필요** | ✅ Oracle Cloud |
| **장기 운영** | ✅ Oracle Cloud |
| **학습/테스트** | 둘 다 OK |

> **이 프로젝트 추천: Oracle Cloud**
> - 이유: 영구 무료, 외부 IP 무료, 한국 리전

---

## 9. Git 브랜치 관리 가이드

### 9.1 브랜치 전략 개요

```
main (메인 브랜치 - 안정 버전)
  │
  └── feature/oracle-cloud-proxy (이 기능 개발용)
        │
        └── 테스트 완료 후 main에 병합
```

### 9.2 현재 브랜치 확인

```bash
# 현재 브랜치 확인
git branch

# 출력:
#   main
# * feature/oracle-cloud-proxy  ← 현재 위치
```

### 9.3 작업 저장 (Commit)

```bash
# 변경된 파일 확인
git status

# 특정 파일 스테이징
git add src/tools/proxy_manager.py
git add config/proxy_config.json
git add docs/guides/ORACLE_CLOUD_PROXY_COMPLETE_GUIDE.md

# 또는 모든 변경사항 스테이징
git add .

# 커밋
git commit -m "feat: Oracle Cloud 프록시 서버 지원 추가

- ProxyManager 클래스 추가
- 프록시 로테이션 지원
- 완전 가이드 문서 추가"
```

### 9.4 임시 저장 (git stash) - 작업 중 급히 다른 것 해야 할 때

```bash
# 현재 작업 임시 저장
git stash

# 저장된 stash 목록 확인
git stash list
# 출력: stash@{0}: WIP on feature/oracle-cloud-proxy: abc1234 메시지

# 다른 브랜치로 이동해서 작업...
git checkout main
# ... 작업 ...
git checkout feature/oracle-cloud-proxy

# 임시 저장한 작업 복원
git stash pop

# 또는 복원하되 stash는 유지
git stash apply
```

**stash 유용한 명령어:**

```bash
# stash에 메시지 추가
git stash push -m "프록시 테스트 중 작업"

# 특정 stash 복원
git stash pop stash@{1}

# stash 삭제
git stash drop stash@{0}

# 모든 stash 삭제
git stash clear
```

### 9.5 브랜치 병합 (테스트 완료 후)

```bash
# 1. main 브랜치로 이동
git checkout main

# 2. 최신 상태로 업데이트
git pull origin main

# 3. feature 브랜치 병합
git merge feature/oracle-cloud-proxy

# 4. 충돌 있으면 해결 후
git add .
git commit -m "merge: Oracle Cloud 프록시 기능 병합"

# 5. 원격에 푸시
git push origin main

# 6. (선택) feature 브랜치 삭제
git branch -d feature/oracle-cloud-proxy
```

### 9.6 병합 전 안전하게 테스트

```bash
# 1. feature 브랜치에서 main의 변경사항 먼저 가져오기
git checkout feature/oracle-cloud-proxy
git merge main

# 2. 충돌 해결 및 테스트
python -m pytest tests/ -v

# 3. 모든 테스트 통과 확인 후 main에 병합
```

### 9.7 실수했을 때 되돌리기

```bash
# 마지막 커밋 취소 (변경사항은 유지)
git reset --soft HEAD~1

# 마지막 커밋 완전 취소 (변경사항도 삭제)
git reset --hard HEAD~1

# 특정 파일만 되돌리기
git checkout -- src/tools/proxy_manager.py

# 브랜치 전체를 원격 상태로 되돌리기
git fetch origin
git reset --hard origin/feature/oracle-cloud-proxy
```

### 9.8 Git 명령어 요약

| 상황 | 명령어 |
|------|--------|
| 새 브랜치 생성 | `git checkout -b feature/xxx` |
| 브랜치 이동 | `git checkout main` |
| 변경사항 저장 | `git add . && git commit -m "메시지"` |
| 임시 저장 | `git stash` |
| 임시 저장 복원 | `git stash pop` |
| 브랜치 병합 | `git merge feature/xxx` |
| 원격 푸시 | `git push origin main` |
| 되돌리기 | `git reset --soft HEAD~1` |

---

## 10. 문제 해결 (Troubleshooting)

### 문제 1: SSH 접속 안 됨

**증상:**
```
ssh: connect to host 129.154.xxx.xxx port 22: Connection timed out
```

**해결:**
1. Oracle Cloud Security List에서 22번 포트 열렸는지 확인
2. VM 상태가 RUNNING인지 확인
3. SSH 키 파일 권한 확인: `chmod 400 your-key.key`

---

### 문제 2: Squid 시작 안 됨

**증상:**
```
Job for squid.service failed
```

**해결:**
```bash
# 설정 오류 확인
sudo squid -k parse

# 로그 확인
sudo journalctl -u squid -n 50
```

---

### 문제 3: 프록시 연결 안 됨

**증상:**
```
curl: (56) Proxy CONNECT aborted
```

**해결:**
1. Security List에서 3128 포트 열렸는지 확인
2. UFW 방화벽 확인: `sudo ufw status`
3. Squid 실행 중인지 확인: `sudo systemctl status squid`

---

### 문제 4: 인증 실패

**증상:**
```
407 Proxy Authentication Required
```

**해결:**
```bash
# 비밀번호 파일 확인
cat /etc/squid/passwd

# 비밀번호 재설정
sudo htpasswd -c /etc/squid/passwd proxyuser

# Squid 재시작
sudo systemctl restart squid
```

---

### 문제 5: Always Free 용량 초과 경고

**증상:**
```
You have exceeded your Always Free limits
```

**해결:**
1. VM Shape이 `VM.Standard.E2.1.Micro`인지 확인
2. 부팅 볼륨이 50GB 이하인지 확인
3. 무료 범위 외 리소스 삭제

---

## 📚 참고 자료

- [Oracle Cloud Free Tier 공식 문서](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm)
- [Oracle Cloud FAQ](https://www.oracle.com/cloud/free/faq/)
- [Squid Proxy 공식 문서](http://www.squid-cache.org/Doc/)
- [Playwright Proxy 설정](https://playwright.dev/python/docs/network#http-proxy)

---

## ✅ 체크리스트

설정 완료 후 확인:

- [ ] Oracle Cloud 계정 생성 완료
- [ ] VM 인스턴스 2개 생성 (proxy-server-1, proxy-server-2)
- [ ] Security List에 3128 포트 추가
- [ ] Squid 설치 및 설정 완료
- [ ] 인증 설정 완료 (username/password)
- [ ] 프록시 테스트 성공 (curl로 IP 확인)
- [ ] Python 코드 연동 완료
- [ ] Git 브랜치 생성 및 커밋

---

**문서 작성일**: 2026-02-02
**버전**: 1.0
**작성자**: Claude AI Assistant
