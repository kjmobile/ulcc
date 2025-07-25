# #num1: Import
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import time
import os
import glob
import pandas as pd

# #num2: 설정
CIRIUM_ID = "dbcobsm2@erau.edu"
CIRIUM_PW = "goEagles2023!#"
download_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data"

# #num3: 공항 리스트
airport_df = pd.read_csv("faa_top100_airports_cy2024.csv")
airports = airport_df['Locid'][:99].tolist()

# #num4: 브라우저 시작
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_path,
    "download.prompt_for_download": False
})
driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 30)

# #num5: 로그인
driver.get("https://mi.diio.net/")
time.sleep(3)

email = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[type='text']")))
email.send_keys(CIRIUM_ID)

password = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
password.send_keys(CIRIUM_PW)

input("로그인 후 Enter...")

# #num6: 메인 루프
for i, origin in enumerate(airports, 1):
    print(f"[{i}/{len(airports)}] {origin}", end=" ")
    
    # #num7: 페이지 로드
    driver.get("https://mi.diio.net/mi/report/OandD/rptOandDMonthlyTraffic.jsp")
    time.sleep(5)
    
    # #num8: Specified Airlines Are를 Segment Marketing Airlines로 설정
    airline_select = Select(wait.until(EC.presence_of_element_located((By.ID, "AIRLINE_ROLE"))))
    airline_select.select_by_value("SEGMENT_MKT")
    
    # #num9: Origin 입력 (원본과 동일한 방식)
    driver.execute_script(f"""
        document.getElementById('ORIGIN_TRIP_SIFTER').value = 'ap!{origin}';
        var originContainer = document.querySelectorAll('.sifter_fieldContainer')[1];
        if(originContainer) {{
            var tagContainer = originContainer.querySelector('.sifter_tagContainer');
            if(tagContainer) {{
                tagContainer.innerHTML = '<div class="sifter_tag airport" title="{origin}">{origin}</div>';
            }}
        }}
    """)
    
    # #num10: 기간 설정
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_PICKER")).select_by_value("VERTICAL_RANGE_OF_MONTHS")
    
    absolute_radio = driver.find_element(By.ID, "TRAVEL_PERIOD_TEMPORAL_ABSOLUTE")
    if not absolute_radio.is_selected():
        absolute_radio.click()
    
    # 시작 날짜 - December 2024
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_MONTH1")).select_by_value("DECEMBER")
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_YEAR1")).select_by_value("2024")
    
    # 종료 날짜 - January 2014  
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_MONTH2")).select_by_value("JANUARY")
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_YEAR2")).select_by_value("2014")
    
    # #num11: 옵션 설정
    # 체크박스 설정 - Operating Airlines만 체크 해제 상태
    checkboxes_config = {
        "SHOW_MKT_AIRLINES": True,
        "SHOW_OP_AIRLINES_TO_MKT_AIRLINE": True,  # 이것도 체크
        "SHOW_SEG_ORIG": True,
        "SHOW_SEG_DEST": True,
        "SHOW_ITINERARY": True,
        "SHOW_PAX_SHARE": True
    }
    
    for checkbox_id, should_check in checkboxes_config.items():
        checkbox = driver.find_element(By.ID, checkbox_id)
        if should_check and not checkbox.is_selected():
            checkbox.click()
        elif not should_check and checkbox.is_selected():
            checkbox.click()
    
    # Show Results를 Per Period로 설정
    Select(driver.find_element(By.ID, "RESULTS_PERIOD")).select_by_value("PERIOD")
    
    # Directionality를 Directional로 설정
    Select(driver.find_element(By.ID, "DIRECTIONALITY")).select_by_value("DIRECTIONAL")
    
    # Level of Detail을 0.0%로 설정
    Select(driver.find_element(By.ID, "LEVEL_OF_DETAIL")).select_by_value("ZERO_PERCENT")
    
    # #num12: 기존 임시 파일만 삭제 (공항 이름의 파일은 보호)
    for f in glob.glob(os.path.join(download_path, "U.S._DOT_O&D_Monthly_Traffic_Report_*.xlsx")):
        try: 
            os.remove(f)
        except: 
            pass
    
    # #num13: 보고서 실행
    Select(driver.find_element(By.ID, "runrep")).select_by_value("formattedPoiExcelReport")
    driver.find_element(By.ID, "runReportLink").click()
    
    # #num14: 다운로드 대기 및 이름 변경
    time.sleep(10)
    downloaded = False
    for wait_count in range(60):  # 60초까지 대기
        # U.S._DOT_O&D_Monthly_Traffic_Report_*.xlsx 패턴 찾기
        files = glob.glob(os.path.join(download_path, "U.S._DOT_O&D_Monthly_Traffic_Report_*.xlsx"))
        
        # 다운로드 완료 확인 (tmp나 crdownload 파일이 없어야 함)
        temp_files = glob.glob(os.path.join(download_path, "*.tmp"))
        crdownload_files = glob.glob(os.path.join(download_path, "*.crdownload"))
        
        if files and not temp_files and not crdownload_files:
            # 가장 최근 파일 찾기
            newest_file = max(files, key=os.path.getctime)
            new_name = os.path.join(download_path, f"{origin}.xlsx")
            
            try:
                # 이미 같은 이름의 파일이 있으면 삭제
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(newest_file, new_name)
                print(f"✓ → {origin}.xlsx")
                downloaded = True
                break
            except Exception as e:
                print(f"✗ 파일명 변경 실패: {e}")
                break
        
        if wait_count % 5 == 0 and wait_count > 0:
            print(".", end="", flush=True)
        time.sleep(1)
    
    if not downloaded:
        print("✗ 다운로드 실패")
    
    time.sleep(5)  # 다음 요청 전 대기

print("\n완료!")

# #num1: Import
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
from datetime import datetime
import gc
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# #num2: 설정
data_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data"
CHUNK_SIZE = 50000

# FAA 공항 확인
faa_df = pd.read_csv('faa_top100_airports_cy2024.csv')
faa_codes = set(faa_df['Locid'].tolist())

# #num3: 날짜 처리
def parse_date(date_value):
    if pd.isna(date_value):
        return None, None
    if isinstance(date_value, (int, float)):
        days = int(date_value) - (1 if int(date_value) > 59 else 0)
        actual_date = datetime(1899, 12, 30) + pd.Timedelta(days=days)
        return actual_date.year, actual_date.month
    try:
        date_obj = pd.to_datetime(date_value)
        return date_obj.year, date_obj.month
    except:
        return None, None

# #num4: 파일 처리
def get_checkpoint():
    """마지막 처리 위치 확인"""
    checkpoint_file = 'airport_checkpoint.txt'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(idx):
    """처리 위치 저장"""
    with open('airport_checkpoint.txt', 'w') as f:
        f.write(str(idx))

def process_files():
    files = glob.glob(os.path.join(data_path, "*.xlsx")) + glob.glob(os.path.join(data_path, "*.csv"))
    file_codes = {os.path.basename(f).split('.')[0] for f in files}
    
    missing = faa_codes - file_codes
    if missing:
        print(f"누락된 공항: {sorted(missing)}")
    
    # 체크포인트 확인
    start_idx = get_checkpoint()
    if start_idx > 0:
        print(f"이전 작업 이어서 진행 (파일 {start_idx+1}부터)")
    
    print(f"처리할 파일: {len(files)}개 (FAA 100개 중 {len(file_codes & faa_codes)}개)")
    
    writers = {}
    cols = ['Dom Op Al', 'Mkt Al 1', 'Org', 'Dst', 'Date', 'Passengers per Period']
    
    for i, filepath in enumerate(files):
        if i < start_idx:
            continue  # 이미 처리된 파일 건너뛰기
            
        airport = os.path.basename(filepath).split('.')[0]
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"\n{i+1}/{len(files)} {airport} ({file_size_mb:.1f}MB) 처리 시작...")
        
        try:
            # 파일 읽기
            print(f"  파일 읽기 중...", end="", flush=True)
            start_time = datetime.now()
            
            if filepath.endswith('.csv'):
                # CSV는 진짜 청크로 읽음
                chunks = pd.read_csv(filepath, skiprows=4, chunksize=CHUNK_SIZE)
                print(" CSV 청크 모드")
                
                # 청크별 처리
                chunk_count = 0
                rows_processed = 0
                for chunk in chunks:
                    chunk_count += 1
                    rows_processed += len(chunk)
                    print(f"\r  청크 {chunk_count} 처리 중... ({rows_processed:,} rows)", end="", flush=True)
                    
                    # 필요한 컬럼만
                    try:
                        chunk = chunk[cols]
                    except KeyError as e:
                        print(f"\n  컬럼 오류: {e}")
                        print(f"  실제 컬럼: {chunk.columns.tolist()[:10]}")
                        continue
                        
                    # 날짜 파싱
                    dates = chunk['Date'].apply(parse_date)
                    chunk['Year'] = dates.apply(lambda x: x[0])
                    chunk['Month'] = dates.apply(lambda x: x[1])
                    
                    # 컬럼명 변경 및 정리
                    chunk = chunk.rename(columns={
                        'Dom Op Al': 'Opr',
                        'Mkt Al 1': 'Mkt',
                        'Passengers per Period': 'Passengers'
                    })
                    chunk = chunk[chunk['Passengers'] > 0].dropna(subset=['Year', 'Month'])
                    
                    # 년도별 저장
                    for year in range(2014, 2025):
                        year_data = chunk[chunk['Year'] == year]
                        if len(year_data) == 0:
                            continue
                        
                        # 타입 최적화
                        year_data = year_data[['Opr', 'Mkt', 'Org', 'Dst', 'Year', 'Month', 'Passengers']].copy()
                        year_data[['Opr', 'Mkt', 'Org', 'Dst']] = year_data[['Opr', 'Mkt', 'Org', 'Dst']].astype(str)
                        year_data['Year'] = year_data['Year'].astype('int16')
                        year_data['Month'] = year_data['Month'].astype('int8')
                        year_data['Passengers'] = year_data['Passengers'].astype('float32')
                        
                        # Parquet 저장
                        output_file = os.path.join(data_path, f'od_{year}.parquet')
                        if year not in writers:
                            table = pa.Table.from_pandas(year_data)
                            writers[year] = pq.ParquetWriter(output_file, table.schema)
                        writers[year].write_table(pa.Table.from_pandas(year_data))
                    
                    gc.collect()
                    
            else:
                # Excel 읽기 - calamine으로 다시 시도
                print("\n  Excel 파일 로딩 중 (calamine)...", end="", flush=True)
                load_start = datetime.now()
                
                df = pd.read_excel(filepath, header=4, engine='calamine')
                
                load_time = datetime.now() - load_start
                print(f" {len(df):,} rows 로드 완료 (소요: {load_time.total_seconds():.1f}초)")
                
                # 데이터 정리
                if 'Passengers per Period' in df.columns:
                    valid_idx = df['Passengers per Period'].notna() & pd.to_numeric(df['Passengers per Period'], errors='coerce').notna()
                    df = df[valid_idx]
                    
                    # 필요한 컬럼만 선택
                    available_cols = [col for col in cols if col in df.columns]
                    if len(available_cols) == len(cols):
                        df = df[cols]
                    else:
                        print(f"\n  경고: 일부 컬럼 누락. 사용 가능: {available_cols}")
                        df = df[available_cols]
                
                # 한번에 처리
                print(f"  데이터 처리 중...", end="", flush=True)
                
                # 날짜 파싱
                dates = df['Date'].apply(parse_date)
                df['Year'] = dates.apply(lambda x: x[0])
                df['Month'] = dates.apply(lambda x: x[1])
                
                # 컬럼명 변경
                df = df.rename(columns={
                    'Dom Op Al': 'Opr',
                    'Mkt Al 1': 'Mkt',
                    'Passengers per Period': 'Passengers'
                })
                df = df[df['Passengers'] > 0].dropna(subset=['Year', 'Month'])
                
                # 년도별 저장
                for year in range(2014, 2025):
                    year_data = df[df['Year'] == year]
                    if len(year_data) == 0:
                        continue
                    
                    year_data = year_data[['Opr', 'Mkt', 'Org', 'Dst', 'Year', 'Month', 'Passengers']].copy()
                    year_data[['Opr', 'Mkt', 'Org', 'Dst']] = year_data[['Opr', 'Mkt', 'Org', 'Dst']].astype(str)
                    year_data['Year'] = year_data['Year'].astype('int16')
                    year_data['Month'] = year_data['Month'].astype('int8')
                    year_data['Passengers'] = year_data['Passengers'].astype('float32')
                    
                    output_file = os.path.join(data_path, f'od_{year}.parquet')
                    if year not in writers:
                        table = pa.Table.from_pandas(year_data)
                        writers[year] = pq.ParquetWriter(output_file, table.schema)
                    writers[year].write_table(pa.Table.from_pandas(year_data))
                    
                gc.collect()
                
            # 파일 처리 완료 후 체크포인트 저장
            save_checkpoint(i + 1)
            print(f"\n  {airport} 완료!")
                
        except Exception as e:
            print(f"\n{airport} 오류: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Writer 닫기
    for writer in writers.values():
        writer.close()
    
    # 완료 시 체크포인트 삭제
    if os.path.exists('airport_checkpoint.txt'):
        os.remove('airport_checkpoint.txt')
    
    print("\n\n=== 결과 ===")
    total_size = 0
    for year in range(2014, 2025):
        filepath = os.path.join(data_path, f'od_{year}.parquet')
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024 / 1024
            total_size += size
            
            # 행 수 확인
            df = pd.read_parquet(filepath)
            print(f"od_{year}.parquet: {size:>6.1f}MB, {len(df):>10,} rows, {df['Org'].nunique():>3} airports")
    
    print(f"\n총 크기: {total_size:.1f}MB")
    print(f"소요 시간: {datetime.now() - start_time}")

# #num5: 실행
start = datetime.now()
process_files()
print(f"\n소요시간: {datetime.now() - start}")

#num24 - LAX 데이터 추가 (날짜 형식 지정)
import pandas as pd
import os
from datetime import datetime

data_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data"
CHUNK_SIZE = 100000

print("LAX 데이터 처리 시작...")
start_time = datetime.now()

all_data = []
cols = ['Dom Op Al', 'Mkt Al 1', 'Org', 'Dst', 'Date', 'Passengers per Period']

chunk_count = 0
for chunk in pd.read_csv(os.path.join(data_path, "LAX.tsv"), 
                        sep='\t', 
                        skiprows=4, 
                        chunksize=CHUNK_SIZE,
                        usecols=cols):
    chunk_count += 1
    
    chunk['Passengers per Period'] = pd.to_numeric(chunk['Passengers per Period'], errors='coerce')
    chunk = chunk[chunk['Passengers per Period'] > 0]
    chunk = chunk[chunk['Date'] != 'TOTAL']
    
    chunk.columns = ['Opr', 'Mkt', 'Org', 'Dst', 'Date', 'Passengers']
    all_data.append(chunk)
    print(f"\r청크 {chunk_count} 처리 중...", end='', flush=True)

print(f"\n총 {chunk_count} 청크 읽기 완료. 데이터 결합 중...")

df = pd.concat(all_data, ignore_index=True)
print(f"총 {len(df):,} rows")

# 날짜 형식 지정으로 빠른 파싱
print("날짜 처리 중...")
df['Date'] = pd.to_datetime(df['Date'], format='%b %Y', errors='coerce')
df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year.astype('int16')
df['Month'] = df['Date'].dt.month.astype('int8')
df['Passengers'] = df['Passengers'].astype('float32')

df = df[['Opr', 'Mkt', 'Org', 'Dst', 'Year', 'Month', 'Passengers']]

# 연도별로 저장
print("연도별 파일 업데이트 중...")
for year in range(2014, 2025):
    year_data = df[df['Year'] == year]
    if len(year_data) == 0:
        continue
    
    output_file = os.path.join(data_path, f'od_{year}.parquet')
    
    if os.path.exists(output_file):
        existing = pd.read_parquet(output_file, engine='pyarrow')
        existing = existing[existing['Org'] != 'LAX']
        combined = pd.concat([existing, year_data], ignore_index=True)
    else:
        combined = year_data
    
    combined.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
    print(f"{year}년: {len(year_data):,} rows 추가됨")

print(f"\nLAX 데이터 추가 완료!")
print(f"소요 시간: {datetime.now() - start_time}")

#num26 - 2024년 parquet 파일 상세 확인
import pandas as pd
import os

data_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data"
filepath = os.path.join(data_path, "od_2024.parquet")

# 파일 읽기
df = pd.read_parquet(filepath)

print("=== 2024년 파일 기본 정보 ===")
print(f"파일 크기: {os.path.getsize(filepath) / 1024 / 1024:.1f}MB")
print(f"총 행 수: {len(df):,}")
print(f"컬럼: {df.columns.tolist()}")
print(f"데이터 타입:\n{df.dtypes}")

print("\n=== 공항 정보 ===")
print(f"총 공항 수: {df['Org'].nunique()}")
print(f"출발 공항 TOP 10:")
top_airports = df['Org'].value_counts().head(10)
for airport, count in top_airports.items():
   print(f"  {airport}: {count:,} rows")

# LAX 데이터 확인
print("\n=== LAX 데이터 확인 ===")
lax_data = df[df['Org'] == 'LAX']
print(f"LAX 총 데이터: {len(lax_data):,} rows")
print(f"LAX 도착 공항 수: {lax_data['Dst'].nunique()}")
print(f"LAX 운항사 수: {lax_data['Opr'].nunique()}")

# 월별 분포
print("\n=== LAX 2024년 월별 데이터 ===")
monthly = lax_data.groupby('Month')['Passengers'].agg(['count', 'sum'])
for month, row in monthly.iterrows():
   print(f"  {month}월: {row['count']:,} 항공편, {row['sum']:,.0f} 승객")

# 데이터 샘플
print("\n=== LAX 데이터 샘플 (처음 5행) ===")
print(lax_data.head())

# 데이터 무결성 확인
print("\n=== 데이터 무결성 확인 ===")
null_check = df.isnull().sum()
print(f"NULL 값 개수:\n{null_check}")

# 년도 확인
print(f"\n연도 확인: {df['Year'].unique()}")

#num27 - 2024년 항공 네트워크 분석
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

data_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data"

# 2024년 데이터 로드
df = pd.read_parquet(os.path.join(data_path, "od_2024.parquet"))

print("=== 2024년 항공 네트워크 기본 분석 ===\n")

# 1. 네트워크 구성
# 공항 간 총 승객 수 집계
edge_data = df.groupby(['Org', 'Dst'])['Passengers'].sum().reset_index()
edge_data = edge_data[edge_data['Passengers'] > 0]

# NetworkX 그래프 생성
G = nx.from_pandas_edgelist(edge_data, 
                            source='Org', 
                            target='Dst', 
                            edge_attr='Passengers',
                            create_using=nx.DiGraph())

print(f"노드 수 (공항): {G.number_of_nodes()}")
print(f"엣지 수 (노선): {G.number_of_edges()}")
print(f"평균 연결도: {G.number_of_edges() / G.number_of_nodes():.2f}")

# 2. 중심성 분석
print("\n=== 공항 중심성 분석 ===")

# Degree Centrality (연결된 공항 수)
degree_cent = nx.degree_centrality(G)
top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n연결도 중심성 TOP 10 (가장 많은 공항과 연결):")
for airport, cent in top_degree:
    connections = len(list(G.neighbors(airport))) + len(list(G.predecessors(airport)))
    print(f"  {airport}: {connections} 개 공항과 연결 (중심성: {cent:.3f})")

# Betweenness Centrality (경유 중심성)
print("\n매개 중심성 TOP 10 (네트워크의 허브 역할):")
between_cent = nx.betweenness_centrality(G, weight='Passengers')
top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:10]
for airport, cent in top_between:
    print(f"  {airport}: {cent:.3f}")

# 3. 승객 수 기준 주요 노선
print("\n=== 주요 노선 TOP 20 (승객 수 기준) ===")
top_routes = edge_data.nlargest(20, 'Passengers')
for _, row in top_routes.iterrows():
    print(f"  {row['Org']} → {row['Dst']}: {row['Passengers']:,.0f} 명")

# 4. LAX 네트워크 분석
print("\n=== LAX 공항 네트워크 분석 ===")
lax_out = [(dest, data['Passengers']) for _, dest, data in G.out_edges('LAX', data=True)]
lax_out_sorted = sorted(lax_out, key=lambda x: x[1], reverse=True)[:10]

print("\nLAX 출발 주요 노선 TOP 10:")
for dest, passengers in lax_out_sorted:
    print(f"  LAX → {dest}: {passengers:,.0f} 명")

# 5. 네트워크 특성
print("\n=== 네트워크 특성 ===")
if nx.is_strongly_connected(G):
    print("강하게 연결된 네트워크: Yes")
else:
    print("강하게 연결된 네트워크: No")
    scc = list(nx.strongly_connected_components(G))
    print(f"강하게 연결된 구성요소 수: {len(scc)}")
    largest_scc = max(scc, key=len)
    print(f"가장 큰 구성요소 크기: {len(largest_scc)} 공항")

# 클러스터링 계수
avg_clustering = nx.average_clustering(G.to_undirected())
print(f"평균 클러스터링 계수: {avg_clustering:.3f}")

# 6. 간단한 시각화 (옵션)
print("\n주요 공항 네트워크 시각화를 생성하시겠습니까? (시간이 걸릴 수 있습니다)")

df.head()

#num30 - 2024년 항공 네트워크 시각화 (타입 경고 수정)
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data"

# 데이터 로드
df = pd.read_parquet(os.path.join(data_path, "od_2024.parquet"))

# 주요 공항만 선택
airport_traffic = df.groupby('Org')['Passengers'].sum().sort_values(ascending=False)
top_airports = airport_traffic.head(30).index.tolist()

# 상위 공항 간의 네트워크만 추출
df_top = df[(df['Org'].isin(top_airports)) & (df['Dst'].isin(top_airports))]
edge_data = df_top.groupby(['Org', 'Dst'])['Passengers'].sum().reset_index()

# NetworkX 그래프 생성
G = nx.from_pandas_edgelist(edge_data, 
                            source='Org', 
                            target='Dst', 
                            edge_attr='Passengers',
                            create_using=nx.DiGraph())

# Figure 1: Network Structure
plt.figure(figsize=(8, 5))

# 노드 크기
node_sizes = []
for node in G.nodes():
    total_passengers = airport_traffic.get(node, 0)
    node_sizes.append(np.sqrt(total_passengers) / 50)

# 노드 색상
degree_cent = nx.degree_centrality(G)
node_colors = [degree_cent[node] for node in G.nodes()]

# 레이아웃
pos = nx.spring_layout(G, k=3, iterations=50)

# 네트워크 그리기
nx.draw_networkx_nodes(G, pos, 
                      node_size=node_sizes,
                      node_color=node_colors,
                      cmap='YlOrRd',
                      alpha=0.8)

# 엣지 그리기
edges = G.edges()
weights = [G[u][v]['Passengers']/1000000 for u, v in edges]
nx.draw_networkx_edges(G, pos, 
                      width=[w*0.5 for w in weights],
                      alpha=0.3,
                      edge_color='gray',
                      arrows=True,
                      arrowsize=10)

# 라벨
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title("2024 US Major 30 Airports Network", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Figure 2: LAX-centric Network
plt.figure(figsize=(8, 5))

lax_neighbors = list(G.successors('LAX')) + list(G.predecessors('LAX')) + ['LAX']
G_lax = G.subgraph(lax_neighbors)

pos_lax = nx.spring_layout(G_lax, center=[0, 0], k=2)
pos_lax['LAX'] = np.array([0, 0])

node_colors_lax = ['red' if node == 'LAX' else 'lightblue' for node in G_lax.nodes()]
node_sizes_lax = [1000 if node == 'LAX' else 300 for node in G_lax.nodes()]

nx.draw_networkx_nodes(G_lax, pos_lax,
                      node_color=node_colors_lax,
                      node_size=node_sizes_lax,
                      alpha=0.8)

edges_lax = G_lax.edges()
weights_lax = []
for u, v in edges_lax:
    if 'Passengers' in G_lax[u][v]:
        weights_lax.append(G_lax[u][v]['Passengers']/500000)
    else:
        weights_lax.append(0.1)

nx.draw_networkx_edges(G_lax, pos_lax,
                      width=weights_lax,
                      alpha=0.5,
                      edge_color='gray',
                      arrows=True)

nx.draw_networkx_labels(G_lax, pos_lax, font_size=10)

plt.title("LAX Airport Direct Flight Network (2024)", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Figure 3: Passenger Flow Heatmap (타입 수정)
plt.figure(figsize=(8, 5))

top20_airports = airport_traffic.head(20).index.tolist()
# float으로 초기화
matrix_data = pd.DataFrame(0.0, index=top20_airports, columns=top20_airports)

for _, row in edge_data.iterrows():
    if row['Org'] in top20_airports and row['Dst'] in top20_airports:
        matrix_data.loc[row['Org'], row['Dst']] = float(row['Passengers'])

import seaborn as sns
sns.heatmap(matrix_data/1000000, 
            cmap='YlOrRd',
            cbar_kws={'label': 'Passengers (Millions)'},
            square=True,
            linewidths=0.5,
            linecolor='white')

plt.title("2024 Airport-to-Airport Passenger Flow Heatmap (Top 20)", fontsize=14)
plt.xlabel("Destination Airport")
plt.ylabel("Origin Airport")
plt.tight_layout()
plt.show()

# Figure 4: Airport Traffic Bar Chart
plt.figure(figsize=(8, 5))

top20_traffic = airport_traffic.head(20)
colors = ['red' if airport == 'LAX' else 'skyblue' for airport in top20_traffic.index]

plt.bar(range(len(top20_traffic)), top20_traffic.values/1000000, color=colors)
plt.xticks(range(len(top20_traffic)), top20_traffic.index, rotation=45)
plt.ylabel("Total Passengers (Millions)")
plt.title("2024 Airport Departure Passenger Traffic TOP 20", fontsize=14)
plt.grid(axis='y', alpha=0.3)

lax_idx = top20_traffic.index.tolist().index('LAX')
plt.text(lax_idx, top20_traffic.iloc[lax_idx]/1000000 + 0.5, 
         'LAX', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("Visualization completed!")

