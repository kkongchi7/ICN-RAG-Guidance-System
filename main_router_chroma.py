"""
main_router_chroma.py
LLM 기반 라우팅
"""

import re, json
from openai import OpenAI

import flights_search_chroma as fr
import facilities_search_chroma as fs
import bus_search_chroma as bs   # ⭐ NEW: 버스 검색 모듈 정식 추가

client = OpenAI()


# ===========================================================
# LLM 기반 Router (FLIGHT / FACILITY / BUS / NONE)
# ===========================================================
def detect_mode_llm(query: str) -> str:
    prompt = f"""
You are a query router for the Incheon Airport assistant.

Classify the user query into EXACTLY one of the following categories:
- FLIGHT
- FACILITY
- BUS
- NONE

Rules:
- Output ONLY one category word.
- No explanation.

User query: "{query}"
"""
    try:
        rsp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        cat = rsp.choices[0].message.content.strip().upper()
    except Exception:
        return "NONE"

    if cat not in ["FLIGHT", "FACILITY", "BUS", "NONE"]:
        return "NONE"
    return cat


# ===========================================================
# LLM 호출 (최종 응답 생성)
# ===========================================================
def ask_llm(prompt: str) -> str:
    rsp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "당신은 인천국제공항 안내 챗봇입니다. 제공된 데이터를 바탕으로 정확하고 간결하게 답하세요."},
            {"role": "user", "content": prompt},
        ],
    )
    return rsp.choices[0].message.content.strip()


# ===========================================================
# 시설 검색 처리
# ===========================================================
def handle_facility_query(query: str, k_fac: int = 6):
    if fs.is_nearby_pattern(query):
        res = fs.structured_nearby_any(query)

        if res.get("error"):
            return {"mode": "nearby", "text": "근처 시설을 찾을 수 없습니다."}

        items = res.get("results", [])
        if not items:
            return {"mode": "nearby", "text": "주변 시설을 찾을 수 없습니다."}

        prompt = fs.build_facility_prompt(query, items)
        return {"mode": "nearby", "text": ask_llm(prompt)}

    hits = fs.search_facilities_chroma(query, k=k_fac)
    prompt = fs.build_facility_prompt(query, hits)
    return {"mode": "facility", "text": ask_llm(prompt)}


# ===========================================================
# 항공편 검색 처리
# ===========================================================
def handle_flight_query(query: str):
    direction = fr.infer_direction(query)
    hits = fr.search_flights_chroma(query, k=10, direction=direction)

    if not hits:
        return {"mode": "flight", "text": "해당 항공편 정보를 찾을 수 없습니다."}

    prompt = fr.build_flight_prompt(query, hits)
    return {"mode": "flight", "text": ask_llm(prompt)}


# ===========================================================
# 체크인 카운터 전용 처리
# ===========================================================
def handle_checkin_counter_query(query: str):
    airline = fr.extract_airline(query)
    if not airline:
        return {"mode": "flight", "text": "어떤 항공사의 체크인 카운터를 찾으시나요?"}

    hits = fr.search_flights_chroma(query, k=30)
    if not hits:
        return {"mode": "flight", "text": f"{airline} 항공편 정보가 없습니다."}

    counters = []
    for r in hits:
        c = (r["meta"].get("체크인 카운터") or "").strip()
        if c and c not in counters:
            counters.append(c)

    terminals = list({r["meta"].get("터미널") for r in hits if r["meta"].get("터미널")})
    terminal_str = ", ".join(terminals) if terminals else "정보 없음"

    ctx = "\n".join([f"- {r['meta'].get('운항편명')}편: {r['meta'].get('체크인 카운터')}" for r in hits[:10]])
    prompt = f"""
아래 항공편 데이터를 참고하여 '{airline}' 항공의 체크인 카운터 정보를 정리해 주세요.

[검색 결과]
{ctx}

요약 규칙:
- 체크인 카운터 위치(A, B, M 등)를 한 문장으로 정리.
- 가능한 경우 터미널({terminal_str})도 함께 언급.
"""

    return {"mode": "flight", "text": ask_llm(prompt)}


# ===========================================================
# 버스 검색 처리
# ===========================================================
def handle_bus_query(query: str, k=5):
    hits = bs.search_bus_routes(query, k=k)

    if not hits:
        return {"mode": "bus", "text": "해당 조건의 공항버스를 찾을 수 없습니다."}

    prompt = bs.build_bus_prompt(query, hits)
    answer = ask_llm(prompt)

    return {"mode": "bus", "text": answer}


# ===========================================================
# 메인 라우터
# ===========================================================
def route_and_answer(query: str, k_fac: int = 4, verbose=False):
    mode = detect_mode_llm(query)

    if verbose:
        print("\n=== 🚀 Chroma RAG Query Start ===")
        print(f"[Router] Mode Detected → {mode}")

    if mode == "FACILITY":
        return handle_facility_query(query, k_fac=k_fac)

    elif mode == "FLIGHT":
        if "체크인" in query or "카운터" in query.lower():
            return handle_checkin_counter_query(query)
        return handle_flight_query(query)

    elif mode == "BUS":
        return handle_bus_query(query)

    else:
        fallback_prompt = f"사용자 질문: {query}\n\n공항 안내 챗봇으로서 적절한 답을 제공하세요."
        return {"mode": "fallback", "text": ask_llm(fallback_prompt)}

