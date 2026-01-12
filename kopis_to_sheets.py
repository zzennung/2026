import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import xml.etree.ElementTree as ET

import gspread
from google.oauth2.service_account import Credentials


# ======================
# ENV (GitHub Actions env로 들어옴)
# ======================
KOPIS_KEY = os.getenv("KOPIS_KEY", "").strip()
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
SHEET_NAME = os.getenv("SHEET_NAME", "performances_raw").strip()

START_DATE = os.getenv("START_DATE", "2025-01-01").strip()
END_DATE = os.getenv("END_DATE", "2026-01-31").strip()

GOOGLE_SERVICE_ACCOUNT = os.getenv("GOOGLE_SERVICE_ACCOUNT", "").strip()

BASE = "http://www.kopis.or.kr/openApi/restful/pblprfr"


def require_env():
    missing = []
    if not KOPIS_KEY:
        missing.append("KOPIS_KEY")
    if not SPREADSHEET_ID:
        missing.append("SPREADSHEET_ID")
    if not GOOGLE_SERVICE_ACCOUNT:
        missing.append("GOOGLE_SERVICE_ACCOUNT")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}. Check GitHub Secrets mapping in workflow yml.")


def chunk_dates(start: str, end: str, max_days: int = 31):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        chunk_end = min(cur + timedelta(days=max_days - 1), e)
        yield cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        cur = chunk_end + timedelta(days=1)


def request_with_retry(url: str, params: dict, retries: int = 5, timeout: int = 30):
    """
    KOPIS가 가끔 불안정할 수 있어서 간단한 재시도(backoff) 추가
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                # 디버깅용 로그
                print("Request URL:", r.url)
                print("Response (first 500 chars):", r.text[:500])
                r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries:
                raise
            sleep_s = min(2 ** attempt, 20)
            print(f"[Retry {attempt}/{retries}] error: {e} -> sleep {sleep_s}s")
            time.sleep(sleep_s)


def fetch_kopis_list(stdate: str, eddate: str, rows: int = 100):
    """
    기간(stdate~eddate) 안의 공연목록을 페이지 끝까지 수집
    stdate/eddate: YYYYMMDD
    """
    results = []
    cpage = 1

    while True:
        params = {
            "service": KOPIS_KEY,
            "stdate": stdate,
            "eddate": eddate,
            "rows": rows,
            "cpage": cpage,
        }

        r = request_with_retry(BASE, params=params, retries=5, timeout=30)

        root = ET.fromstring(r.text)
        dbs = root.findall(".//db")
        if not dbs:
            break

        for db in dbs:
            mt20id = (db.findtext("mt20id") or "").strip()
            prfnm = (db.findtext("prfnm") or "").strip()
            fcltynm = (db.findtext("fcltynm") or "").strip()

            if mt20id and prfnm:
                results.append({"공연ID": mt20id, "공연명": prfnm, "공연장": fcltynm})

        cpage += 1

    return results


def get_gspread_client():
    """
    Secret 'GOO
