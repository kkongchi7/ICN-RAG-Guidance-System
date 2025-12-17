"""
build_spatial_index_2.py
공항 시설 JSON에서 공간 인덱스 생성
"""

import json, math
from pathlib import Path

# =========================
# ⚙️ 설정값
# =========================
INPUT_JSON = "/content/spoi_formatted_with_category.json"
OUTPUT_JSON = "/content/spatial_index.json"

MODE = "h3"        # h3 또는 grid
H3_RES = 12        # H3 해상도 (높을수록 정밀)
GRID_SIZE_M = 100  # grid 모드일 경우 셀 크기 (m)

# =========================
# H3 or Grid 로직
# =========================
try:
    import h3
except ImportError:
    h3 = None

def build_index_h3(items):
    index = []
    for fac in items:
        lat, lon = fac.get("poiLatitude"), fac.get("poiLongitude")
        if not lat or not lon:
            continue
        try:
            cell = h3.geo_to_h3(lat, lon, H3_RES)
        except Exception:
            cell = h3.latlng_to_cell(lat, lon, H3_RES)
        index.append({
            "id": fac["vsid"],
            "building": fac.get("building"),
            "floor": fac.get("floor"),
            "cell": cell,
        })

    grouped = {}
    for e in index:
        k = (e["building"], e["floor"], e["cell"])
        grouped.setdefault(k, []).append(e["id"])

    keys = [
        {"building": b, "floor": f, "cell": c, "ids": ids}
        for (b, f, c), ids in grouped.items()
    ]
    return {
        "mode": "h3",
        "h3_res": H3_RES,
        "keys": keys
    }

def build_index_grid(items):
    all_lat = [f["poiLatitude"] for f in items if f.get("poiLatitude")]
    all_lon = [f["poiLongitude"] for f in items if f.get("poiLongitude")]
    if not all_lat or not all_lon:
        raise ValueError("좌표 데이터가 부족합니다.")

    min_lat, max_lat = min(all_lat), max(all_lat)
    min_lon, max_lon = min(all_lon), max(all_lon)
    ref_lat = (min_lat + max_lat) / 2

    M_PER_DEG_LAT = 111320.0
    M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(ref_lat))

    origin = {"lat": min_lat, "lon": min_lon}

    def cell_for(lat, lon):
        dx_m = (lat - origin["lat"]) * M_PER_DEG_LAT
        dy_m = (lon - origin["lon"]) * M_PER_DEG_LON
        gx = int(round(dx_m / GRID_SIZE_M))
        gy = int(round(dy_m / GRID_SIZE_M))
        return f"{gx}:{gy}"

    index = []
    for fac in items:
        lat, lon = fac.get("poiLatitude"), fac.get("poiLongitude")
        if not lat or not lon:
            continue
        cell = cell_for(lat, lon)
        index.append({
            "id": fac["vsid"],
            "building": fac.get("building"),
            "floor": fac.get("floor"),
            "cell": cell,
        })

    grouped = {}
    for e in index:
        k = (e["building"], e["floor"], e["cell"])
        grouped.setdefault(k, []).append(e["id"])

    keys = [
        {"building": b, "floor": f, "cell": c, "ids": ids}
        for (b, f, c), ids in grouped.items()
    ]
    return {
        "mode": "grid",
        "grid_size_m": GRID_SIZE_M,
        "origin": origin,
        "ref_lat_deg": ref_lat,
        "keys": keys
    }

# =========================
# 실행부
# =========================
if __name__ == "__main__":
    print(f"=== 🧭 Spatial Index Builder ({MODE.upper()} mode) ===")
    data = json.loads(Path(INPUT_JSON).read_text())
    items = data.get("items", data)
    print(f"Loaded {len(items)} facilities")

    if MODE == "h3":
        idx = build_index_h3(items)
    else:
        idx = build_index_grid(items)

    Path(OUTPUT_JSON).write_text(json.dumps(idx, ensure_ascii=False, indent=2))
    print(f"✅ spatial_index.json saved → {OUTPUT_JSON}")
    print(f"mode={idx['mode']} | #keys={len(idx['keys'])}")
    print("sample:", idx["keys"][0] if idx["keys"] else None)

