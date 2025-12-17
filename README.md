# ✈️ ICN-RAG: Incheon International Airport RAG System

## Project Summary

본 프로젝트는 **인천국제공항(Incheon International Airport, ICN)** 이용자를 위한  
**RAG(Retrieval-Augmented Generation) 기반 대화형 정보 검색 시스템**이다.

사용자의 자연어 질의를 분석하여  
- ✈️ 항공편 정보 (출발 / 도착 / 날짜 / 항공사)
- 🏢 공항 시설 정보 (카페, 편의시설, 매장 등)
- 🧭 위치 기반 검색 (게이트·시설 기준 “근처” 질의)
- 🚌 공항버스 노선 및 시간표 정보  

중 하나로 자동 라우팅한 뒤,  
**ChromaDB 기반 벡터 검색 + 메타데이터 필터링**을 통해 정확한 정보를 제공한다.

본 시스템은 실제 공항 환경에서의 **복합 질의 처리**,  
**불완전한 사용자 발화**,  
**공간적 맥락(같은 터미널·층·근처)**을 고려한 검색을 목표로 설계되었다.

---

### 1️⃣ Environment Setup (Google Colab 기준)

본 프로젝트는 **Google Colab 환경**에서 실행하는 것을 기준으로 작성되었다.

#### 📌 1. Google Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')

#### 📌 2. 필수 라이브러리 설치
pip install chromadb sentence-transformers h3 pandas numpy

#### 📌 3. 기본 경로 설정
import sys

BASE_DIR = "/content/drive/MyDrive/Colab Notebooks"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

#### 📌 4. 임베딩 모델 로드
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cuda"
)

#### 📌 5. ChromaDB 초기화
import chromadb

chroma_client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="/content/chroma_db",
        anonymized_telemetry=False
    )
)

---

### 2️⃣ Project Structure
📦 ICN-RAG
 ┣ 📜 ICN_RAG_Guidance_System_vColab.ipynb   # 메인 실행 노트북 (Colab)
 ┣ 📜 README.md
 ┣ 📜 main_router_chroma.py                 # LLM + Rule 기반 질의 라우터
 ┣ 📜 flights_search_chroma.py              # 항공편 검색 모듈
 ┣ 📜 facilities_search_chroma.py           # 공항 시설 검색 모듈
 ┣ 📜 bus_search_chroma.py                  # 공항버스 검색 모듈
 ┣ 📜 build_spatial_index_2.py               # H3 기반 공간 인덱스 생성
 ┣ 📜 city_airline_map.py                   # 도시 / 항공사 매핑
 ┣ 📜 facility_category_map.py              # 시설 카테고리 매핑

---

### 3️⃣ Execution Flow
1. ICN_RAG_Guidance_System_vColab.ipynb 실행

2. 사용자 자연어 질의 입력

3. main_router_chroma.py
→ 질의 유형 분류 (FLIGHT / FACILITY / BUS)

4. 도메인별 검색 모듈 호출

5. ChromaDB 벡터 검색 + 메타데이터 필터링

6. 최종 응답 출력


## DEMO
### 🧪 Example Queries
내일 김포 가는 대한항공 항공편 알려줘
250번 게이트 근처 카페 찾아줘
야탑 가는 공항버스 몇 시야?
제2터미널 3층 화장실 어디야?

### 🔍 Routing Example
[Router] Mode Detected → FACILITY
[Anchor] 250번 게이트
[Target] 카페

### ✅ Output Example
A: 인천국제공항 제1여객터미널(T1) 3층에서 추천드릴 만한 카페로 “커피앳웍스”를 안내해 드립니다!
시설명: 커피앳웍스
제1여객터미널(T1) 3층, 12번 출입구 부근
영업시간: 06:00 ~ 22:00
전화번호: 032-743-3776
취급품목: 신선한 커피와 다양한 디저트

## 🚀 Future Work

📍 실내 지도 기반 경로 탐색 (Gate → Facility)

🗺️ 지도 시각화 UI 연동

🔊 음성 입력 (STT) 및 음성 응답 (TTS)

🌐 다국어 질의 응답 고도화

📊 검색 성능 정량 평가 (Recall@K, MRR)

🧠 다중 턴 질의 및 맥락 유지
