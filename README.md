# ✈️ ICN-RAG: Incheon International Airport RAG System

본 프로젝트는 **인천국제공항(Incheon International Airport, ICN)** 이용자를 위한 **RAG(Retrieval-Augmented Generation) 기반 대화형 정보 검색 시스템**입니다. 

사용자의 자연어 질의를 분석하여 항공편, 공항 시설, 공항버스 정보를 자동으로 라우팅하고, **ChromaDB 기반 벡터 검색**과 **H3 공간 인덱싱**을 결합하여 최적의 정보를 제공합니다.

---

## 🌟 Key Features

* **질의 Routing**: LLM과 Rule 기반 로직을 활용하여 질의를 FLIGHT / FACILITY / BUS 카테고리로 자동 분류.
* **공간 기반 검색**: H3 라이브러리를 통해 게이트나 특정 시설 "근처"의 위치 기반 검색 지원.
* **복합 필터링**: ChromaDB의 메타데이터 필터링을 사용하여 터미널, 층, 항공사 등 상세 조건 검색.
* **한국어 최적화**: `multilingual-e5-base` 모델을 사용하여 한국어 자연어 이해도 향상.

---

## 🛠️ Environment Setup (Google Colab)

이 프로젝트는 Google Colab 환경에서 실행되도록 설계되었습니다.

### 1. 라이브러리 설치
```python
!pip install chromadb sentence-transformers h3 pandas numpy

```

### 2. 구글 드라이브 마운트 및 경로 설정

```python
from google.colab import drive
import sys

drive.mount('/content/drive')

# 프로젝트 경로 설정 (본인의 경로에 맞게 수정)
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

```

### 3. 모델 및 DB 초기화

```python
from sentence_transformers import SentenceTransformer
import chromadb

# 임베딩 모델 로드
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")

# ChromaDB 초기화
chroma_client = chromadb.PersistentClient(path="/content/chroma_db")

```

---

## 📂 Project Structure

```text
📦 ICN-RAG
 ┣ 📜 ICN_RAG_Guidance_System_vColab.ipynb  # 메인 실행 노트북
 ┣ 📜 main_router_chroma.py                # 질의 유형 분류 라우터
 ┣ 📜 flights_search_chroma.py             # 항공편 검색 모듈
 ┣ 📜 facilities_search_chroma.py          # 공항 시설 검색 모듈
 ┣ 📜 bus_search_chroma.py                 # 공항버스 검색 모듈
 ┣ 📜 build_spatial_index_2.py             # H3 기반 공간 인덱스 생성
 ┣ 📜 city_airline_map.py                  # 도시/항공사 매핑 데이터
 ┗ 📜 facility_category_map.py             # 시설 카테고리 매핑 데이터

```

---

## ⚙️ System Architecture
<img width="1492" height="772" alt="image" src="https://github.com/user-attachments/assets/9770fa30-1eff-45e8-99cc-e4ad0ced6af0" />

1. **User Query**: "250번 게이트 근처 카페 알려줘"
2. **Router**: 질의를 분석하여 `FACILITY` 모드로 분류 및 키워드(카페) 추출.
3. **Retriever**: ChromaDB에서 250번 게이트 좌표 기준 반경 내 '카페' 카테고리 검색.
4. **Generator**: 검색된 메타데이터(영업시간, 위치, 전화번호)를 바탕으로 자연어 답변 생성.

---

## 🧪 Demo & Examples

| 질의 유형 | 예시 질문 |
| --- | --- |
| **항공편** | "내일 도쿄 가는 대한항공 항공편 알려줘" |
| **시설 검색** | "제2터미널 3층 화장실 어디야?" |
| **근처 검색** | "250번 게이트 근처 카페 추천해줘" |
| **공항버스** | "야탑 가는 공항버스 첫차 시간 알려줘" |

### ✅ Output Example

> **Q: 250번 게이트 근처 카페 찾아줘**
> **[Router]** Mode: FACILITY | **Anchor**: 250번 게이트 | **Target**: 카페
> **A:** 인천국제공항 제1여객터미널(T1) 3층, 250번 게이트 부근에 있는 **'커피앳웍스'**를 추천합니다. (영업시간: 06:00~22:00)

---

## 🚀 Future Work

* 📍 **Indoor Navigation**: 게이트에서 시설까지의 실제 보행 경로 시각화.
* 🌐 **Multilingual Support**: 영어, 중국어, 일본어 등 다국어 대응 고도화.
* 🧠 **Multi-turn Context**: 이전 대화의 맥락을 기억하는 대화형 엔진 강화.
* 📊 **Evaluation**: RAGAS를 활용한 응답 정확성 정량 평가.

---
