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
download_path = r"G:\My Drive\3_1 network_scott\gnns_in_action\play\data\t_100"

# 폴더 생성
os.makedirs(download_path, exist_ok=True)

# #num3: 공항 리스트
airport_df = pd.read_csv("faa_top100_airports_cy2024.csv")
airports = airport_df['Locid'][:100].tolist()  # 100개 공항

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
    
    # #num7: 페이지 새로고침 (처음 공항이 아닌 경우)
    if i > 1:
        driver.refresh()
        time.sleep(5)
    else:
        # 첫 번째 공항일 때만 페이지 이동
        driver.get("https://mi.diio.net/mi/report/T100/rptT100Summary.jsp")
        time.sleep(5)
    
    # 페이지 로드 완료 대기
    wait.until(EC.presence_of_element_located((By.ID, "ORIGIN_TRIP_SIFTER")))
    
    # #num8: Summary 옵션 선택 (드롭다운에서)
    # T-100 Summary가 기본값이므로 별도 선택 불필요
    
    # #num9: Origin 입력 (O&D와 동일한 방식)
    driver.execute_script(f"""
        document.getElementById('ORIGIN_TRIP_SIFTER').value = 'ap!{origin}';
        var originContainer = document.querySelectorAll('.sifter_fieldContainer')[2];
        if(originContainer) {{
            var tagContainer = originContainer.querySelector('.sifter_tagContainer');
            if(tagContainer) {{
                tagContainer.innerHTML = '<div class="sifter_tag airport" title="{origin}">{origin}</div>';
            }}
        }}
    """)
    
    # #num10: 기간 설정 - Monthly Time Series
    travel_period = Select(driver.find_element(By.ID, "TRAVEL_PERIOD_PICKER"))
    travel_period.select_by_value("VERTICAL_RANGE_OF_MONTHS")
    
    # Absolute Dates 선택 (이미 기본값으로 선택되어 있음)
    absolute_radio = driver.find_element(By.ID, "TRAVEL_PERIOD_TEMPORAL_ABSOLUTE")
    if not absolute_radio.is_selected():
        absolute_radio.click()
    
    # 시작 날짜 - 2014년 1월
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_MONTH1")).select_by_visible_text("January")
    Select(driver.find_element(By.ID, "TRAVEL_PERIOD_YEAR1")).select_by_visible_text("2014")
    
    # 종료 날짜 - 2024년 12월
    # 먼저 종료 날짜 필드 찾기
    month2_elements = driver.find_elements(By.XPATH, "//select[contains(@name, 'MONTH') and contains(@id, '2')]")
    year2_elements = driver.find_elements(By.XPATH, "//select[contains(@name, 'YEAR') and contains(@id, '2')]")
    
    if month2_elements and year2_elements:
        Select(month2_elements[0]).select_by_visible_text("December")
        Select(year2_elements[0]).select_by_visible_text("2024")
    else:
        # 대체 방법: name 속성으로 찾기
        Select(driver.find_element(By.NAME, "rp[TRAVEL_PERIOD_MONTH2]")).select_by_visible_text("December")
        Select(driver.find_element(By.NAME, "rp[TRAVEL_PERIOD_YEAR2]")).select_by_visible_text("2024")
    
    # #num11: 체크박스 및 옵션 설정
    # Marketing Airlines 체크 (이미 체크됨)
    marketing_cb = driver.find_element(By.ID, "SHOW_MKT_AIRLINES")
    if not marketing_cb.is_selected():
        marketing_cb.click()
    
    # Operating Airlines 체크 해제
    operating_cb = driver.find_element(By.ID, "SHOW_OP_AIRLINES_TO_MKT_AIRLINE")
    if operating_cb.is_selected():
        operating_cb.click()
    
    # Origins 체크
    origins_cb = driver.find_element(By.ID, "SHOW_SEG_ORIG")
    if not origins_cb.is_selected():
        origins_cb.click()
    
    # Destinations 체크
    destinations_cb = driver.find_element(By.ID, "SHOW_SEG_DEST")
    if not destinations_cb.is_selected():
        destinations_cb.click()
    
    # Aircraft Details 체크
    aircraft_cb = driver.find_element(By.ID, "SHOW_AIRCRAFT_DETAILS")
    if not aircraft_cb.is_selected():
        aircraft_cb.click()
    
    # Directionality 설정 - DIRECTIONAL
    Select(driver.find_element(By.ID, "DIRECTIONALITY")).select_by_value("DIRECTIONAL")
    
    # Show Results - Per Period
    Select(driver.find_element(By.ID, "RESULTS_PERIOD")).select_by_value("PERIOD")
    
    # Service Type - Sched. Passenger
    Select(driver.find_element(By.ID, "T100_SERVICE_TYPE")).select_by_visible_text("Sched. Passenger")
    
    # Only Show Deps/Month >= 1
    Select(driver.find_element(By.ID, "MIN_DEPS_PER_MONTH")).select_by_value("ONE")
    
    # #num12: 기존 임시 파일 삭제 (T-100 패턴의 TSV 파일)
    for f in glob.glob(os.path.join(download_path, "U.S._DOT_T-100_*.tsv")):
        try: 
            os.remove(f)
        except: 
            pass
    
    # #num13: 보고서 실행 - TSV 형식으로
    # 드롭다운에서 TSV 선택
    report_dropdown = Select(driver.find_element(By.ID, "runrep"))
    report_dropdown.select_by_visible_text("TSV")
    
    # Run Report 링크 클릭
    driver.find_element(By.ID, "runReportLink").click()
    
    # #num14: 다운로드 대기 및 이름 변경
    time.sleep(10)
    downloaded = False
    for wait_count in range(60):
        # U.S._DOT_T-100 패턴의 TSV 파일 찾기
        files = glob.glob(os.path.join(download_path, "U.S._DOT_T-100_*.tsv"))
        temp_files = glob.glob(os.path.join(download_path, "*.tmp"))
        crdownload_files = glob.glob(os.path.join(download_path, "*.crdownload"))
        
        if files and not temp_files and not crdownload_files:
            newest_file = max(files, key=os.path.getctime)
            new_name = os.path.join(download_path, f"T100_{origin}.tsv")
            
            try:
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(newest_file, new_name)
                print(f"✓ → T100_{origin}.tsv")
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
    
    time.sleep(5)

print("\n완료!")

# #num1 - 초기 설정 및 라이브러리 import
import pandas as pd
import os
import glob

# #num2 - 경로 및 설정값 정의
INPUT_FOLDER = 'data/t_100'
OUTPUT_FOLDER = 'data/t_100'
FAA_PATH = 'faa_top100_airports_cy2024.csv'

# #num3 - FAA 공항 코드 로드
faa_df = pd.read_csv(FAA_PATH)
faa_codes = set(faa_df['Locid'].astype(str).str.strip())

# #num4 - 컬럼명 정의 (모든 컬럼 포함)
columns = ['Mkt Al', 'Orig', 'Dest', 'Miles', 'Date', 'Aircraft Config', 
          'Aircraft Group', 'Aircraft Type', 'Deps', 'Deps/Day', 
          'Onboards', 'Seats', 'RPMs', 'ASMs', 'Load Factor']

# #num5 - 월 이름 매핑
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# #num6 - TSV 파일 목록
tsv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, 'T100_*.tsv')))
print(f"Found {len(tsv_files)} TSV files")

# #num7 - 연도별 처리
for year in range(2014, 2025):
   year_data = []
   
   # #num8 - 파일별 처리
   for path in tsv_files:
       try:
           # #num9 - 청크 단위로 읽기
           chunks = pd.read_csv(path, sep='\t', skiprows=6, names=columns, 
                              chunksize=10000, dtype={'Date': str, 'Mkt Al': str})
           
           for chunk in chunks:
               # #num10 - 데이터 정제
               chunk['Date'] = chunk['Date'].astype(str)
               chunk = chunk[chunk['Date'] != 'nan']
               chunk = chunk[~chunk['Date'].str.contains('TOTAL', na=False)]
               
               # #num11 - FAA 공항 필터링
               chunk['Orig'] = chunk['Orig'].astype(str)
               chunk['Dest'] = chunk['Dest'].astype(str)
               chunk = chunk[chunk['Orig'].isin(faa_codes) & chunk['Dest'].isin(faa_codes)]
               
               if len(chunk) == 0:
                   continue
               
               # #num12 - 날짜 파싱 및 연도 필터링
               date_parts = chunk['Date'].str.extract(r'([A-Za-z]{3})\s+(\d{4})')
               chunk['Year'] = pd.to_numeric(date_parts[1], errors='coerce')
               chunk['Month'] = date_parts[0].map(month_map)
               
               chunk = chunk[chunk['Year'].notna() & chunk['Month'].notna()]
               chunk = chunk[chunk['Year'] == year]
               
               if len(chunk) == 0:
                   continue
               
               # #num13 - 숫자형 변환 (모든 숫자 컬럼)
               numeric_cols = ['Miles', 'Deps', 'Deps/Day', 'Onboards', 'Seats', 'RPMs', 'ASMs', 'Load Factor']
               chunk[numeric_cols] = chunk[numeric_cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))
               
               # #num14 - 모든 컬럼 유지 (Date 제외)
               keep_cols = ['Mkt Al', 'Orig', 'Dest', 'Year', 'Month', 'Miles', 
                          'Aircraft Config', 'Aircraft Group', 'Aircraft Type',
                          'Deps', 'Deps/Day', 'Onboards', 'Seats', 'RPMs', 'ASMs', 'Load Factor']
               year_data.append(chunk[keep_cols])
               
       except Exception as e:
           print(f"❌ {os.path.basename(path)}: {e}")
           continue
   
   # #num15 - 연도별 집계 및 저장 (Mkt Al 포함)
   if year_data:
       df_year = pd.concat(year_data, ignore_index=True)
       
       # Mkt Al 포함하여 그룹핑
       df_agg = df_year.groupby(['Mkt Al', 'Orig', 'Dest', 'Year', 'Month']).agg({
           'Miles': 'first',
           'Aircraft Config': 'first',
           'Aircraft Group': 'first', 
           'Aircraft Type': 'first',
           'Deps': 'sum',
           'Deps/Day': 'mean',
           'Onboards': 'sum',
           'Seats': 'sum',
           'RPMs': 'sum',
           'ASMs': 'sum',
           'Load Factor': 'mean'
       }).reset_index()
       
       output_path = os.path.join(OUTPUT_FOLDER, f't_100_{year}.parquet')
       df_agg.to_parquet(output_path, compression='snappy')
       print(f"✅ {year}: {len(df_agg):,} rows → {output_path}")
       
       del df_year, df_agg
   
   year_data = []

# #num1 - T-100 집계 검증 스크립트
import pandas as pd

# #num2 - 2024년 파일로 검증
df = pd.read_parquet('data/t_100/t_100_2024.parquet')

print("=== 파일 정보 ===")
print(f"Shape: {df.shape}")
print(f"\n컬럼 목록: {df.columns.tolist()}")

# #num3 - 데이터 타입 확인
print(f"\n=== 데이터 타입 ===")
print(df.dtypes)

# #num4 - 샘플 데이터
print(f"\n=== 샘플 데이터 (5행) ===")
print(df.head())

# #num5 - 그룹핑 키 확인
grouping_keys = ['Mkt Al', 'Orig', 'Dest', 'Year', 'Month']
print(f"\n=== 그룹핑 키 분석 ===")

# 유니크 조합 수
unique_combinations = df[grouping_keys].drop_duplicates()
print(f"유니크한 키 조합 수: {len(unique_combinations):,}")

# #num6 - 집계된 컬럼들 분석
print(f"\n=== 집계 방식 확인 ===")
numeric_cols = ['Miles', 'Deps', 'Deps/Day', 'Onboards', 'Seats', 'RPMs', 'ASMs', 'Load Factor']

for col in numeric_cols:
   if col in df.columns:
       print(f"{col}: 존재 ✓")
   else:
       print(f"{col}: 없음 ✗")

# #num7 - 특정 항공사-노선 예시
print(f"\n=== 예시: AA의 ORD-LAX 노선 ===")
example = df[(df['Mkt Al'] == 'AA') & (df['Orig'] == 'ORD') & (df['Dest'] == 'LAX')]
if len(example) > 0:
   print(example[['Month', 'Deps', 'Seats', 'Onboards']].head())
else:
   # 다른 예시 찾기
   sample_airline = df['Mkt Al'].value_counts().index[0]
   sample_route = df[df['Mkt Al'] == sample_airline].iloc[0]
   print(f"\nAA ORD-LAX 없음. 대신 {sample_airline}의 {sample_route['Orig']}-{sample_route['Dest']} 예시:")
   example = df[(df['Mkt Al'] == sample_airline) & 
                (df['Orig'] == sample_route['Orig']) & 
                (df['Dest'] == sample_route['Dest'])]
   print(example[['Month', 'Deps', 'Seats', 'Onboards']].head())

# #num8 - 항공사별 통계
print(f"\n=== 상위 10개 항공사 ===")
airline_stats = df.groupby('Mkt Al')['Onboards'].sum().sort_values(ascending=False).head(10)
print(airline_stats)

# #num9 - 중복 확인
duplicates = df.duplicated(subset=grouping_keys)
print(f"\n=== 중복 체크 ===")
print(f"중복된 행 수: {duplicates.sum()}")
if duplicates.sum() > 0:
   print("⚠️ 경고: 동일한 키 조합이 여러 번 나타남!")
else:
   print("✅ 좋음: 각 키 조합이 유니크함")

