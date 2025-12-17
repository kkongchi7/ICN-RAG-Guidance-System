# -*- coding: utf-8 -*- 
"""
bus_search_chroma.py
인천공항 버스 노선 검색 (LLM을 이용한 목적지 추출 + ChromaDB 검색)
"""

import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import time
import csv
import openai
import ast
import pandas as pd

openai_client = openai.OpenAI()

# ===== 설정 =====
# (이전에 성공했던 설정값과 일치시켜서 DB를 로드합니다)
CHROMA_PATH = "./chroma_bus_db" 
COLLECTION_NAME = "bus_routes"
MODEL_NAME = "intfloat/multilingual-e5-base"


# ChromaDB 클라이언트 설정
# NOTE: 이미 임베딩 시점에 E5 모델로 벡터가 생성되었으므로,
# 단순히 컬렉션을 가져옵니다.
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME) 
model = SentenceTransformer(MODEL_NAME)

# =======================
# 🚍 LLM을 통한 목적지 추출
# =======================
def extract_destination_from_query(query: str) -> str:
    """
    LLM을 이용해 질의에서 목적지 추출
    """
    prompt = f"""
    아래 사용자 질의에서 목적지로 추측되는 장소를 추출해주세요. 
    주의사항:
    - 정확한 장소만 추출하세요. 추가적인 추측은 금지입니다.
    
    사용자 질문: "{query}"
    목적지: 
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0,
        n=1,
        stop=None
    )

    destination = response.choices[0].message.content.strip()
    return destination


# =======================
# 📃 CSV 파일에서 버스 노선 불러오기 (파싱 로직 통합)
# =======================
def load_bus_data():
    """
    CSV 파일에서 버스 노선 데이터를 로드하고 리스트 문자열을 파싱합니다.
    """
    try:
        # 파일 경로는 첨부된 파일에서 지정된 경로를 따릅니다.
        with open("/content/airport_bus_routes.csv", newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            bus_data = [row for row in reader]
        
        # 데이터 파싱: 문자열 리스트를 실제 Python 리스트로 변환
        list_cols = ['T1_weekday', 'T1_weekend', 'T2_weekday', 'T2_weekend', 'stops']
        
        parsed_data = []
        for row in bus_data:
            new_row = row.copy()
            for col in list_cols:
                if col in new_row and new_row[col]:
                    try:
                        # 안전하게 문자열 리스트를 평가
                        new_row[col] = ast.literal_eval(new_row[col])
                    except:
                        new_row[col] = [] # 파싱 실패 시 빈 리스트
            parsed_data.append(new_row)
            
        return parsed_data
    except Exception as e:
        print(f"Error loading and parsing bus data: {e}")
        return []

# =======================
# 🧠 임베딩 처리 (버스 정보) - Passage Generation Logic Integrated
# =======================
def create_bus_embeddings():
    """
    버스 노선 정보를 임베딩하여 ChromaDB에 저장합니다.
    (이전에 성공했던 descriptive document structure를 따르며 E5 prefix를 추가합니다)
    """
    bus_data = load_bus_data()
    documents = []
    metadatas = []
    ids = []

    for idx, row in enumerate(bus_data):
        
        # 1. Descriptive Passage Generation (이전에 성공한 긴 문장 구조)
        region = row.get("region", "")
        bus_no = row.get("bus_no", "")
        route_id = row.get("route_id", "")
        
        # load_bus_data를 통해 이미 리스트로 파싱된 데이터 사용
        stops = row.get("stops", []) 
        T1_weekday = row.get("T1_weekday", [])
        
        stops_str = ", ".join(stops) if stops else "정보 없음"
        t1_times = ", ".join(T1_weekday[:5]) # 시간표는 앞 5개만 간략하게 포함
        
        doc_text = (
            # E5 모델 접두사 'passage:' 추가
            f"passage: {region} 지역 공항버스 {bus_no}번 노선 정보입니다. "
            f"노선 ID는 {route_id}입니다. "
            f"주요 경유 정류장은 {stops_str} 입니다. "
            f"인천공항 제1터미널 평일 첫차 시간대는 {t1_times} 등 입니다. "
            f"제1터미널 탑승 위치는 {row.get('boarding_T1', '정보 없음')}, "
            f"제2터미널 탑승 위치는 {row.get('boarding_T2', '정보 없음')} 입니다."
        )
        documents.append(doc_text)
        
        # 2. Metadata Preparation (ChromaDB 호환: 리스트는 문자열로 변환)
        metadata = row.copy()
        list_cols = ['T1_weekday', 'T1_weekend', 'T2_weekday', 'T2_weekend', 'stops']
        
        for col in list_cols:
            if col in metadata and isinstance(metadata[col], list):
                # 리스트를 파이프(|)로 구분된 문자열로 변환하여 저장
                metadata[col] = "|".join(metadata[col]) 
            elif col in metadata and (pd.isna(metadata[col]) or metadata[col] is None):
                metadata[col] = "" # None/NaN 처리
        
        metadatas.append(metadata)
        
        # 3. ID Generation (이전에 사용된 결합 형식 사용)
        ids.append(f"{region}_{bus_no}_{idx}") 

    # E5 모델은 접두사가 이미 doc_text에 포함되어 있습니다.
    embeddings = model.encode(documents, normalize_embeddings=True)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"총 {len(documents)}개의 버스 노선 임베딩 저장 완료 (Descriptive Text & E5 prefix).")


# =======================
# 🧭 ChromaDB 검색 (목적지 기반) - Faulty Metadata Filter 제거
# =======================
def search_bus_routes(query: str, k=5):
    """
    LLM을 통해 목적지를 추출하고, ChromaDB에서 해당 목적지로 가는 버스 노선 검색
    (Metadata 필터 대신 Semantic Search에 전적으로 의존합니다.)
    """
    destination = extract_destination_from_query(query)
    print(f"목적지 추출됨: {destination}")

    # 추출된 목적지를 쿼리에 포함하고, E5 모델의 'query:' 접두사를 사용
    query_text = f"query: {query} {destination}" if destination else f"query: {query}"

    query_embedding = model.encode([query_text], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=query_embedding,
        # where=where_clause, # 메타데이터 필터 미사용
        n_results=k
    )

    # 검색된 결과 처리
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist
        hits.append({"score": round(score, 4), "text": doc, "meta": meta})

    return hits


# =======================
# 📝 LLM 프롬프트 생성 (수정된 메타데이터 구조 반영)
# =======================
def build_bus_prompt(query: str, hits: list):
    """
    LLM 프롬프트 생성: 검색된 버스 정보를 바탕으로 응답
    """
    if not hits:
        return f"'{query}'에 해당하는 버스 정보를 찾을 수 없습니다."

    cards = []
    for h in hits[:5]:
        meta = h["meta"]
        bus_no = meta.get("bus_no", "-")
        # 메타데이터의 stops는 이제 "|"로 join된 문자열이므로 그대로 사용합니다.
        stops_str = meta.get("stops", "-") 
        T1_weekday = meta.get("T1_weekday", "-")
        T1_weekend = meta.get("T1_weekend", "-")
        T2_weekday = meta.get("T2_weekday", "-")
        T2_weekend = meta.get("T2_weekend", "-")
        
        cards.append(
            f"- 버스 {bus_no}번 | 정류장: {stops_str} | T1 평일: {T1_weekday} | T1 주말: {T1_weekend} | T2 평일: {T2_weekday} | T2 주말: {T2_weekend}"
        )

    ctx = "\n".join(cards)
    prompt = f"""
    당신은 인천국제공항 안내 챗봇입니다. 
    아래 데이터를 참고하여 사용자 질문에 친절하고 활기차게 답변하세요.
    - 여러 노선이 있을 경우 요약하여 답변하세요.
    - 데이터에 없는 내용은 추측하지 마세요. 모르는 정보는 언급하지 마세요.
    - 답변은 반드시 사용자 질문에 충실하게 답해주세요.

    사용자 질문: "{query}"

    [검색된 버스 정보]
    {ctx}
    """
    return prompt


# =======================
# 🏢 버스 검색 처리 (라우터 유지)
# =======================
def search_bus_chroma(query: str, k=5):
    """
    버스 검색 및 LLM 응답 처리
    """
    hits = search_bus_routes(query, k=k)

    if not hits:
        return {"mode": "bus", "text": "해당 조건의 공항버스를 찾을 수 없습니다."}

    prompt = build_bus_prompt(query, hits)
    answer = ask_llm(prompt)

    return {"mode": "bus", "text": answer}


def ask_llm(prompt: str) -> str:
    """
    LLM을 사용하여 최종 답변을 생성
    """
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# NOTE: 이 파일은 모듈로 작동하므로, __name__ == "__main__" 부분은 제거했습니다.