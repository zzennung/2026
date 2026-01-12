import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import xml.etree.ElementTree as ET

import gspread
from google.oauth2.service_account import Credentials


# ======================
# ENV
# ======================
KOPIS_KEY = os.getenv("KOPIS_KEY", "").strip()
if not KOPIS_KEY:
    raise RuntimeError("KOPIS_KEY is missing. Check GitHub Secrets mapping.")

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
if not SPREADSHEET_ID:
    raise RuntimeError("SPREADSHEET_ID is missing. Check GitHub Secrets mapping.")

SHEET_NAME = os.environ.get("SHEET_NAME", "performances_raw")
START_DATE = os.environ.get("START_DATE", "2025-01-01")
END_DATE = os.environ.get("END_DATE", "2026-01-31")

# KOPIS endpoint
BASE = "http://www.kopis.or.kr/openApi/restful/pblprfr"


def chunk_dates(start: str, end: str, max_days=31):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        chunk_end = min(cur + timedelta(days=max_days - 1), e)
        yield cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        cur = chunk_end + timedelta(days=1)


def fetch_kopis_list(stdate: str, eddate: str, rows=100):
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

        r = requests.get(BASE, params=params, timeout=30)

        # 에러 났을 때 원인 파악 쉽게 로그
        if r.status_code >= 400:
            print("Request URL:", r.url)
            print("Response (first 500 chars):", r.text[:500])
            r.raise_for_status()

        root = ET.fromstring(r.text)
        dbs = root.findall(".//db")
        if not dbs:
            break

        for db in dbs:
            mt20id = (db.findtext("mt20id") or "").strip()
            prfnm = (db.findtext("prfnm") or "").strip()
            fcltynm = (db.findtext("fcltynm") or "").strip()

            if mt20id and prfnm:
                results.append(
                    {
                        "공연ID": mt20id,
                        "공연명": prfnm,
                        "공연장": fcltynm,
                    }
                )

        cpage += 1

    return results


def get_gspread_client():
    """
    GitHub Secrets에 'GOOGLE_SERVICE_ACCOUNT' 이름으로 저장된
    서비스계정 JSON(원문 문자열)을 그대로 읽어 인증합니다.
    """
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT", "").strip()
    if not sa_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT is missing. Add it to GitHub Secrets and pass via workflow env.")
