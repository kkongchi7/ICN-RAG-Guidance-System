"""
bus_search_chroma.py
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
CHROMA_PATH = "./chroma_bus_db" 
COLLECTION_NAME = "bus_routes"
MODEL_NAME = "intfloat/multilingual-e5-base"


# ChromaDB 클라이언트 설정
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME) 
model = SentenceTransformer(MODEL_NAME)

# =======================
# LLM을 통한 목적지 추출
# =======================
def extract_destination_from_query(query: str) -> str:
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
# CSV 파일에서 버스 노선 불러오기 (파싱 로직 통합)
# =======================
def load_bus_data():
    try:
        with open("/content/airport_bus_routes.csv", newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            bus_data = [row for row in reader]
        
        list_cols = ['T1_weekday', 'T1_weekend', 'T2_weekday', 'T2_weekend', 'stops']
        
        parsed_data = []
        for row in bus_data:
            new_row = row.copy()
            for col in list_cols:
                if col in new_row and new_row[col]:
                    try:
                        new_row[col] = ast.literal_eval(new_row[col])
                    except:
                        new_row[col] = []
            parsed_data.append(new_row)
            
        return parsed_data
    except Exception as e:
        print(f"Error loading and parsing bus data: {e}")
        return []

# =======================
# 임베딩 처리 (버스 정보)
# =======================
def create_bus_embeddings():
    bus_data = load_bus_data()
    documents = []
    metadatas = []
    ids = []

    for idx, row in enumerate(bus_data):
        
        region = row.get("region", "")
        bus_no = row.get("bus_no", "")
        route_id = row.get("route_id", "")
        
        stops = row.get("stops", []) 
        T1_weekday = row.get("T1_weekday", [])
        
        stops_str = ", ".join(stops) if stops else "정보 없음"
        t1_times = ", ".join(T1_weekday[:5]) 
        
        doc_text = (
            f"passage: {region} 지역 공항버스 {bus_no}번 노선 정보입니다. "
            f"노선 ID는 {route_id}입니다. "
            f"주요 경유 정류장은 {stops_str} 입니다. "
            f"인천공항 제1터미널 평일 첫차 시간대는 {t1_times} 등 입니다. "
            f"제1터미널 탑승 위치는 {row.get('boarding_T1', '정보 없음')}, "
            f"제2터미널 탑승 위치는 {row.get('boarding_T2', '정보 없음')} 입니다."
        )
        documents.append(doc_text)
        
        metadata = row.copy()
        list_cols = ['T1_weekday', 'T1_weekend', 'T2_weekday', 'T2_weekend', 'stops']
        
        for col in list_cols:
            if col in metadata and isinstance(metadata[col], list):
                metadata[col] = "|".join(metadata[col]) 
            elif col in metadata and (pd.isna(metadata[col]) or metadata[col] is None):
                metadata[col] = "" # None/NaN 처리
        
        metadatas.append(metadata)
        
        ids.append(f"{region}_{bus_no}_{idx}") 

    embeddings = model.encode(documents, normalize_embeddings=True)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"총 {len(documents)}개의 버스 노선 임베딩 저장 완료 (Descriptive Text & E5 prefix).")


# =======================
# ChromaDB 검색
# =======================
def search_bus_routes(query: str, k=5):
    destination = extract_destination_from_query(query)
    print(f"목적지 추출됨: {destination}")

    query_text = f"query: {query} {destination}" if destination else f"query: {query}"

    query_embedding = model.encode([query_text], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist
        hits.append({"score": round(score, 4), "text": doc, "meta": meta})

    return hits


# =======================
# LLM 프롬프트 생성
# =======================
def build_bus_prompt(query: str, hits: list):
    if not hits:
        return f"'{query}'에 해당하는 버스 정보를 찾을 수 없습니다."

    cards = []
    for h in hits[:5]:
        meta = h["meta"]
        bus_no = meta.get("bus_no", "-")
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
# 버스 검색 처리
# =======================
def search_bus_chroma(query: str, k=5):
    hits = search_bus_routes(query, k=k)

    if not hits:
        return {"mode": "bus", "text": "해당 조건의 공항버스를 찾을 수 없습니다."}

    prompt = build_bus_prompt(query, hits)
    answer = ask_llm(prompt)

    return {"mode": "bus", "text": answer}


def ask_llm(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
