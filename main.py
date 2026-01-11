import os
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import xml.etree.ElementTree as ET
import time

# [설정] KOPIS API 및 구글 인증
KOPIS_API_KEY = os.environ.get('KOPIS_API_KEY')
GOOGLE_SERVICE_ACCOUNT = os.environ.get('GOOGLE_SERVICE_ACCOUNT')
SPREADSHEET_ID = "1EtJvJapMlMjxTqRWyWimzk7EKxW9cGBOVKQEZF4jbsU"

# Github Actions에서 입력받은 시설명 (기본값: 예술의전당)
FACILITY_NAME = os.environ.get('FACILITY_NAME', '예술의전당')
HALL_NAME = os.environ.get('HALL_NAME')  # 예: "올림픽홀"

def get_facility_id(name):
    """3.1 공연시설 목록 조회 API를 사용하여 시설 ID 획득 [cite: 24, 27]"""
    url = "http://www.kopis.or.kr/openApi/restful/prfplc"
    params = {
        'service': KOPIS_API_KEY,
        'cpage': 1,
        'rows': 10,
        'shprfnmfct': name  # 공연시설명 [cite: 27]
    }
    res = requests.get(url, params=params)
    root = ET.fromstring(res.content)
    for db in root.findall('db'):
        if db.findtext('fcltynm') == name:
            return db.findtext('mt10id')  # 공연시설 ID 반환 [cite: 32]
    return None

def get_performance_details(mt20id):
    """2.1 공연 상세 조회 API를 사용하여 가격 및 주최 정보 획득 [cite: 15, 22]"""
    url = f"http://www.kopis.or.kr/openApi/restful/pblprfr/{mt20id}"
    params = {'service': KOPIS_API_KEY}
    res = requests.get(url, params=params)
    root = ET.fromstring(res.content)
    db = root.find('db')
    if db is not None:
        # 주최(entrpsnmH), 주관(entrpsnmS) 정보 추출 [cite: 22]
        host = db.findtext('entrpsnmH') or ""
        manager = db.findtext('entrpsnmS') or ""
        return {
            "티켓가격": db.findtext('pcseguidance') or "가격정보 없음", # [cite: 22]
            "주최/주관": f"{host} / {manager}".strip(" / ") or "정보 없음"
        }
    return {"티켓가격": "-", "주최/주관": "-"}

def get_performance_list(fclty_id):
    """1.1 공연목록 조회 API를 사용하여 기간 내 공연 수집 [cite: 3, 7]"""
    url = "http://www.kopis.or.kr/openApi/restful/pblprfr"
    # 조회 기간 설정: 2025.01.01 ~ 2026.01.31
    stdate = "20250101"
    eddate = "20260131"
    
    params = {
        'service': KOPIS_API_KEY,
        'stdate': stdate,
        'eddate': eddate,
        'cpage': 1,
        'rows': 100, # 최대 100건 [cite: 8]
        'prfplccd': fclty_id # 공연시설 ID 필터 [cite: 8]
    }
    
    res = requests.get(url, params=params)
    root = ET.fromstring(res.content)
    
    results = []
    days_map = ['월', '화', '수', '목', '금', '토', '일']
    
    for db in root.findall('db'):
        mt20id = db.findtext('mt20id') # 공연 ID [cite: 14]
        prfnm = db.findtext('prfnm')   # 공연명 [cite: 14]
        p_start = db.findtext('prfpdfrom') # 공연 시작일 [cite: 14]
        
        # 요일 계산 (YYYY.MM.DD 형식 파싱)
        dt = datetime.strptime(p_start, '%Y.%m.%d')
        date_with_day = f"{p_start}({days_map[dt.weekday()]})"
        
        # 상세 정보 수집 (가격, 주최)
        details = get_performance_details(mt20id)
        time.sleep(0.1) # API 매너 지연
        
        results.append({
            "공연명": prfnm,
            "공연일자(요일포함)": date_with_day,
            "티켓가격": details['티켓가격'],
            "주최/주관": details['주최/주관']
        })
    return pd.DataFrame(results)

def save_to_sheet(df, facility_name):
    """스프레드시트에 새 탭 생성 및 저장"""
    creds = Credentials.from_service_account_info(
        json.loads(GOOGLE_SERVICE_ACCOUNT), 
        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    )
    client = gspread.authorize(creds)
    sh = client.open_by_key(SPREADSHEET_ID)
    
    # 새 탭 이름 (시설명_조회시간)
    tab_name = f"{facility_name}_{datetime.now().strftime('%m%d_%H%M')}"
    worksheet = sh.add_worksheet(title=tab_name, rows=len(df)+5, cols=5)
    
    # 상단 헤더 및 데이터 추가
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
    print(f"✅ 저장 완료: '{tab_name}' 탭")

if __name__ == "__main__":
    print(f"🔍 '{FACILITY_NAME}' 시설 검색 중...")
    f_id = get_facility_id(FACILITY_NAME)
    
    if f_id:
        print(f"📍 시설 ID 확인: {f_id}. 공연 데이터를 가져옵니다...")
        df = get_performance_list(f_id)
        if not df.empty:
            save_to_sheet(df, FACILITY_NAME)
        else:
            print("❌ 해당 기간 내 공연 데이터가 없습니다.")
    else:
        print(f"⚠️ '{FACILITY_NAME}' 시설을 찾을 수 없습니다. 명칭을 확인해 주세요.")
