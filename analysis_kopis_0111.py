import os
import re
import math
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


BASE = "http://www.kopis.or.kr/openApi/restful"
LIST_ENDPOINT = f"{BASE}/pblprfr"             # 공연목록
DETAIL_ENDPOINT = f"{BASE}/pblprfr"           # 공연상세: /pblprfr/{mt20id}

# KOPIS 공통코드: shcate=CCCD → 대중음악
SHCATE_POP = "CCCD"


def ymd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def daterange_chunks(start: datetime, end: datetime, max_days: int = 31):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=max_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def safe_text(elem, tag: str) -> str:
    node = elem.find(tag)
    return node.text.strip() if node is not None and node.text else ""


def fetch_list(service_key: str, start: datetime, end: datetime, rows: int = 100) -> pd.DataFrame:
    """
    pblprfr 공연목록: stdate~eddate 최대 31일 제한 (가이드 기준)
    """
    all_rows = []
    for s, e in daterange_chunks(start, end, max_days=31):
        cpage = 1
        while True:
            params = {
                "service": service_key,
                "stdate": ymd(s),
                "eddate": ymd(e),
                "cpage": cpage,
                "rows": rows,
                "shcate": SHCATE_POP,  # 대중음악만
            }
            r = requests.get(LIST_ENDPOINT, params=params, timeout=30)
            r.raise_for_status()

            root = ET.fromstring(r.text)
            dbs = root.findall(".//db")

            if not dbs:
                break

            for db in dbs:
                all_rows.append({
                    "mt20id": safe_text(db, "mt20id"),
                    "prfnm": safe_text(db, "prfnm"),
                    "genrenm": safe_text(db, "genrenm"),
                    "prfstate": safe_text(db, "prfstate"),
                    "prfpdfrom": safe_text(db, "prfpdfrom"),
                    "prfpdto": safe_text(db, "prfpdto"),
                    "fcltynm": safe_text(db, "fcltynm"),
                    "area": safe_text(db, "area"),
                    "openrun": safe_text(db, "openrun"),
                    "poster": safe_text(db, "poster"),
                })

            # pagesize보다 적게 오면 마지막 페이지로 간주
            if len(dbs) < rows:
                break
            cpage += 1

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["mt20id"])
    return df


def fetch_detail(service_key: str, mt20id: str) -> dict:
    """
    공연상세: /pblprfr/{mt20id}?service=...
    """
    url = f"{DETAIL_ENDPOINT}/{mt20id}"
    params = {"service": service_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    db = root.find(".//db")
    if db is None:
        return {}

    return {
        "mt20id": mt20id,
        "mt10id": safe_text(db, "mt10id"),
        "prfcast": safe_text(db, "prfcast"),
        "prfcrew": safe_text(db, "prfcrew"),
        "prfruntime": safe_text(db, "prfruntime"),
        "prfage": safe_text(db, "prfage"),
        "entrpsnmP": safe_text(db, "entrpsnmP"),
        "entrpsnmA": safe_text(db, "entrpsnmA"),
        "entrpsnmH": safe_text(db, "entrpsnmH"),
        "entrpsnmS": safe_text(db, "entrpsnmS"),
        "pcseguidance": safe_text(db, "pcseguidance"),  # 티켓가격 텍스트
        "dtguidance": safe_text(db, "dtguidance"),
        "visit": safe_text(db, "visit"),
        "festival": safe_text(db, "festival"),
        "updatedate": safe_text(db, "updatedate"),
        "sty": safe_text(db, "sty"),
    }


def parse_date_kopis(s: str) -> datetime | None:
    # "2025.11.01" 형태
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y.%m.%d")
    except ValueError:
        return None


def extract_price_krw(text: str) -> float:
    """
    pcseguidance에서 금액(원) 숫자만 대충 뽑아 대표값(중앙값)을 만든다.
    예: "전석 30,000원" / "R석 110,000원 S석 99,000원" 등
    """
    if not isinstance(text, str) or not text.strip():
        return float("nan")
    nums = re.findall(r"(\d[\d,]{2,})\s*원", text)
    vals = []
    for n in nums:
        try:
            vals.append(int(n.replace(",", "")))
        except:
            pass
    if not vals:
        return float("nan")
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2)


def _as_str(x) -> str:
    # None / NaN(float) 안전 처리
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

def primary_subgenre(prfnm: str, prfcast: str, sty: str) -> str:
    """
    “대중음악 내” 세부 포맷(룰 기반) 1차 태깅.
    *원하면 너 회사 기준으로 룰 더 촘촘히 맞춰줄 수 있음.
    """
    text = " ".join([
        _as_str(prfnm),
        _as_str(prfcast),
        _as_str(sty),
    ]).lower()

    rules = [
        ("virtual", r"(버츄얼|virtual|vtuber|v-tuber|브이튜버|이세계아이돌|홀로라이브|니지산지)"),
        ("hiphop", r"(힙합|hiphop|rapper|랩|cypher|싸이퍼|크루|crew)"),
        ("electronic_dj", r"(dj|edm|electronic|techno|house|trance|dnb|drum\s*and\s*bass)"),
        ("ost_ip", r"(ost|게임|game|애니|anime|지브리|final\s*fantasy|젤다|닌텐도)"),
        ("rnb_soul", r"(r&b|rnb|soul|알앤비|소울)"),
        ("jpop_kpop_idol", r"(k-pop|케이팝|아이돌|fan\s*meeting|팬미팅|j-pop|jpop)"),
        ("band", r"(band|밴드|라이브밴드|guitar|드럼|bass|베이스)"),
        ("singer_songwriter", r"(싱어송라이터|singer-songwriter|어쿠스틱|acoustic)"),
    ]
    for label, pat in rules:
        if re.search(pat, text, flags=re.IGNORECASE):
            return label
    return "other_pop"



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 날짜
    df["start_dt"] = df["prfpdfrom"].apply(parse_date_kopis)
    df["end_dt"] = df["prfpdto"].apply(parse_date_kopis)
    df["run_days"] = (df["end_dt"] - df["start_dt"]).dt.days + 1
    df.loc[df["run_days"] < 1, "run_days"] = 1

    # 가격(대표값)
    df["ticket_price_krw"] = df["pcseguidance"].apply(extract_price_krw)

    # 플래그
    df["is_openrun"] = df["openrun"].astype(str).str.upper().eq("Y").astype(int)
    df["is_visit"] = df["visit"].astype(str).str.upper().eq("Y").astype(int)
    df["is_festival"] = df["festival"].astype(str).str.upper().eq("Y").astype(int)

    # 텍스트 기반 세부장르
    df["subgenre"] = df.apply(lambda r: primary_subgenre(r.get("prfnm"), r.get("prfcast"), r.get("sty")), axis=1)

    # 지역/공연장 다양성(공연 단위라서 “단순 정보”로만)
    df["has_area"] = df["area"].fillna("").ne("").astype(int)

    # 클러스터링에 쓸 feature (결측은 중앙값/0으로 처리)
    feats = ["run_days", "ticket_price_krw", "is_openrun", "is_visit", "is_festival", "has_area"]
    for c in feats:
        if c in ["ticket_price_krw"]:
            df[c] = df[c].astype(float)
        else:
            df[c] = df[c].fillna(0)

    df["ticket_price_krw"] = df["ticket_price_krw"].fillna(df["ticket_price_krw"].median())
    df["run_days"] = df["run_days"].fillna(1)

    return df


def cluster_and_plot(df: pd.DataFrame, k: int = 6, outdir: str = "outputs"):
    os.makedirs(outdir, exist_ok=True)

    feats = ["run_days", "ticket_price_krw", "is_openrun", "is_visit", "is_festival", "has_area"]
    X = df[feats].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Xs)

    df_out = df.copy()
    df_out["cluster"] = labels
    df_out.to_csv(os.path.join(outdir, "popmusic_clustered.csv"), index=False, encoding="utf-8-sig")

    # ---- 시각화 1: 대표 포지셔닝 (러닝기간 vs 티켓가격) ----
    plt.figure()
    plt.scatter(df_out["run_days"], df_out["ticket_price_krw"], c=df_out["cluster"])
    plt.xlabel("Run days (기간 일수)")
    plt.ylabel("Ticket price (KRW, 대표값)")
    plt.title("KOPIS Pop Music (CCCD) — Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_run_days_vs_price.png"), dpi=180)
    plt.close()

    # ---- 시각화 2: 클러스터별 세부장르 분포(상위) ----
    top = (
        df_out.groupby(["cluster", "subgenre"])["mt20id"]
        .count()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )
    top.to_csv(os.path.join(outdir, "cluster_subgenre_counts.csv"), index=False, encoding="utf-8-sig")

    # 클러스터별 상위 subgenre만 보기 쉽게 요약
    summary = (
        top.groupby("cluster")
        .apply(lambda g: g.head(5))
        .reset_index(drop=True)
    )
    summary.to_csv(os.path.join(outdir, "cluster_subgenre_top5.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] Saved: {outdir}/popmusic_clustered.csv")
    print(f"[OK] Saved: {outdir}/scatter_run_days_vs_price.png")
    print(f"[OK] Saved: {outdir}/cluster_subgenre_top5.csv")


def main():
    service_key = os.environ.get("KOPIS_API_KEY", "").strip()
    if not service_key:
        raise RuntimeError("환경변수 KOPIS_API_KEY가 비어있어요. (export KOPIS_API_KEY=... 먼저)")

    # 최근 3년: 2023-01-01 ~ 2025-12-31 (현재 대화 기준)
    start = datetime(2023, 1, 1)
    end = datetime(2025, 12, 31)

    print("[1/4] Fetch list (pblprfr, CCCD=대중음악) ...")
    df_list = fetch_list(service_key, start, end, rows=100)
    print(f"  - list rows: {len(df_list):,}")

    print("[2/4] Fetch detail for each performance id ... (시간 좀 걸릴 수 있음)")
    details = []
    for mt20id in tqdm(df_list["mt20id"].tolist()):
        try:
            details.append(fetch_detail(service_key, mt20id))
        except Exception:
            # API가 간헐적으로 실패할 수 있어, 일단 스킵하고 진행
            continue

    df_detail = pd.DataFrame(details)
    print(f"  - detail rows: {len(df_detail):,}")

    print("[3/4] Merge + build features ...")
    df = df_list.merge(df_detail, on="mt20id", how="left")
    df = build_features(df)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/popmusic_raw_enriched.csv", index=False, encoding="utf-8-sig")

    print("[4/4] Clustering + plots ...")
    cluster_and_plot(df, k=6, outdir="outputs")


if __name__ == "__main__":
    main()
