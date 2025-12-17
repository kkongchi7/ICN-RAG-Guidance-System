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

## Code Instruction

### 1️⃣ Environment Setup

```bash
pip install chromadb sentence-transformers h3 pandas numpy
