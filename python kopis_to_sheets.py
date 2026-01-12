import os
import base64
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from dateutil.parser import parse
import xml.etree.ElementTree as ET

import gspread
from google.oauth2.service_account import Credentials


# ======================
# ENV
# ======================
KOPIS_KEY = os.environ["KOPIS_KEY"]
SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
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
    # GitHub Secrets에 base64로 넣은 서비스계정 JSON을 복원
    sa_b64 = os.environ["GOOGLE_SA_JSON_B64"]
    sa_json = base64.b64decode(sa_b64).decode("utf-8")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(eval(sa_json) if sa_json.strip().startswith("{") is False else __import__("json").loads(sa_json), scopes=scopes)
    return gspread.authorize(creds)


def upsert_sheet(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame):
    gc = get_gspread_client()
    sh = gc.open_by_key(spreadsheet_id)

    try:
        ws = sh.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=10)

    ws.clear()

    values = [df.columns.tolist()] + df.fillna("").values.tolist()
    ws.update(values)


def main():
    all_rows = []
    chunks = list(chunk_dates(START_DATE, END_DATE, max_days=31))

    for st, ed in tqdm(chunks, desc="Fetching"):
        all_rows.extend(fetch_kopis_list(st, ed, rows=100))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("KOPIS returned no rows. Check dates/key.")

    df = df.drop_duplicates(subset=["공연ID"]).reset_index(drop=True)
    df_out = df[["공연명", "공연장"]].copy()

    upsert_sheet(SPREADSHEET_ID, SHEET_NAME, df_out)
    print(f"Done. Rows uploaded: {len(df_out)}")


if __name__ == "__main__":
    main()
