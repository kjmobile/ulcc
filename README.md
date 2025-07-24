#  Code Map

## 전체 프로젝트 구조
```
airline_analysis/
├── main_analysis.py          # 메인 컨트롤러
├── run_analysis.py           # 실행 스크립트
├── data_loader.py            # 데이터 로딩
├── network_analysis.py       # H2 네트워크 분석
├── market_behavior_analysis.py # H1 시장 행동 분석
├── competition_analysis.py   # H3 경쟁 영향 분석
├── covid_recovery_analysis.py # H4 COVID 회복 분석
├── visualization_manager.py  # 시각화 관리
└── report/                   # 결과 파일들
```

---

## 파일별 상세 맵

### 1. `main_analysis.py` - 메인 컨트롤러
**클래스**: `AirlineAnalysisController`
**주요 속성**:
- `self.data_loader`: 데이터 로더 인스턴스
- `self.results`: 분석 결과 저장

**주요 메서드**:
- `run_complete_analysis()`: 전체 분석 실행
- `setup_analysis()`: 초기 설정
- `generate_final_report()`: 최종 보고서 생성

**중요 변수**:
- `analysis_results`: 각 가설별 결과 딕셔너리
- `start_time`: 분석 시작 시간

---

### 2. `data_loader.py` - 데이터 로딩 및 전처리
**클래스**: `AirlineDataLoader`
**주요 속성**:
- `self.od_data`: Origin-Destination 데이터
- `self.t100_data`: T-100 운항 데이터
- `self.shock_data`: 외부 충격 변수 데이터
- `self.carrier_classification`: 항공사 분류

**주요 메서드**:
- `load_all_data()`: 모든 데이터 로딩
- `classify_carriers()`: 항공사 비즈니스 모델 분류
- `prepare_competition_data()`: 경쟁 분석용 데이터 준비

**중요 변수**:
- `CARRIER_TYPES`: {'ULCC', 'LCC', 'Hybrid', 'Legacy'}
- `data_paths`: 데이터 파일 경로 딕셔너리

---

### 3. `network_analysis.py` - H2 네트워크 구조 분석
**클래스**: `NetworkAnalyzer`
**주요 속성**:
- `self.od_data`: OD 데이터
- `self.carrier_classification`: 항공사 분류
- `self.carrier_metrics`: 항공사별 네트워크 지표

**주요 메서드**:
- `analyze_network_structure()`: 네트워크 구조 분석
- `calculate_modularity()`: 모듈러리티 계산
- `test_h2_hypothesis()`: H2 가설 검증

**중요 변수**:
- `Modularity`: 네트워크 모듈러리티 점수
- `Gini`: 허브 집중도 지니계수
- `Top3_Hub_Share`: 상위 3개 허브 점유율
- `Density`: 네트워크 밀도

---

### 4. `market_behavior_analysis.py` - H1 시장 행동 분석
**클래스**: `MarketBehaviorAnalyzer`
**주요 속성**:
- `self.od_data`: OD 데이터
- `self.route_behavior`: 노선별 진입/철수 행동
- `self.carrier_dynamics`: 항공사별 시장 역학

**주요 메서드**:
- `analyze_market_behavior()`: 시장 행동 분석
- `calculate_entry_exit_rates()`: 진입/철수율 계산
- `test_h1_hypothesis()`: H1 가설 검증

**중요 변수**:
- `Entry_Rate`: 시장 진입률
- `Exit_Rate`: 시장 철수율
- `Route_Churn`: 노선 변동률
- `Retention_Rate`: 노선 유지율

---

### 5. `competition_analysis.py` - H3 경쟁 영향 분석
**클래스**: `CompetitionAnalyzer`
**주요 속성**:
- `self.competition_data`: 경쟁 구조 데이터
- `self.instrumental_variables`: 도구변수 데이터
- `self.regression_results`: 회귀분석 결과

**주요 메서드**:
- `analyze_competition_effects()`: 경쟁 효과 분석
- `run_2sls_analysis()`: 2단계 최소제곱법 분석
- `test_h3_hypothesis()`: H3 가설 검증

**중요 변수**:
- `HHI`: 허핀달-허쉬만 지수 (시장집중도)
- `ULCC_Share`: ULCC 시장점유율
- `Load_Factor`: 탑승률
- `IV_Hist_ULCC`: ULCC 과거 도구변수

---

### 6. `covid_recovery_analysis.py` - H4 COVID 회복 분석
**클래스**: `CovidRecoveryAnalyzer`
**주요 속성**:
- `self.shock_data`: 외부 충격 데이터
- `self.recovery_metrics`: 회복 지표
- `self.market_share_evolution`: 시장점유율 변화

**주요 메서드**:
- `analyze_covid_impact()`: COVID 영향 분석
- `calculate_recovery_speed()`: 회복 속도 계산
- `test_h4_hypothesis()`: H4 가설 검증

**중요 변수**:
- `Recovery_Months`: 90% 회복까지 소요 개월
- `Trough_Performance`: 최저점 성과 (2019년 대비 %)
- `Market_Share_Change`: 시장점유율 변화 (2019→2023)
- `COVID_Dummy`: COVID 기간 더미변수

---

### 7. `visualization_manager.py` - 시각화 관리
**클래스**: `VisualizationManager`
**주요 속성**:
- `self.fig_count`: Figure 번호 카운터
- `self.output_dir`: 출력 디렉토리 ('report/')

**주요 메서드**:
- `create_hypothesis_figures()`: 가설별 Figure 생성
- `create_summary_dashboard()`: 종합 대시보드 생성
- `save_figure()`: Figure 저장

**중요 변수**:
- `figure_specs`: Figure 사양 딕셔너리
- `color_palette`: 항공사 타입별 색상 팔레트

---

## 데이터 플로우

```
데이터 파일들 (parquet)
    ↓
data_loader.py (데이터 로딩 & 전처리)
    ↓
main_analysis.py (분석 조율)
    ↓
각 분석 모듈들 (H1, H2, H3, H4)
    ↓
visualization_manager.py (시각화)
    ↓
report/ 폴더 (최종 결과물)
```

---

## 주요 데이터 구조

### 항공사 분류 (CARRIER_TYPES)
```python
{
    'ULCC': ['NK', 'F9', 'G4'],      # Spirit, Frontier, Allegiant
    'LCC': ['WN', 'FL', 'SY'],       # Southwest, AirTran, Sun Country
    'Hybrid': ['AS', 'B6', 'HA', 'VX'], # Alaska, JetBlue, Hawaiian, Virgin
    'Legacy': ['AA', 'DL', 'UA', 'US']   # American, Delta, United, US Airways
}
```

### 데이터 테이블 구조
**OD Data**: `['Opr', 'Mkt', 'Org', 'Dst', 'Year', 'Month', 'Passengers']`
**T-100 Data**: `['Mkt Al', 'Orig', 'Dest', 'Year', 'Month', 'ASMs', 'RPMs', 'Load Factor', ...]`
**Shock Data**: `['WTI_Price', 'JetFuel_Price', 'COVID_Dummy', 'Workplace_Mobility', ...]`

---

## 가설별 핵심 변수

### H1 (Market Behavior)
- **종속변수**: `Entry_Rate`, `Exit_Rate`, `Route_Churn`
- **독립변수**: `Carrier_Type`
- **예상 순서**: ULCC > LCC > Hybrid > Legacy

### H2 (Network Structure)  
- **종속변수**: `Modularity`
- **독립변수**: `Carrier_Type`
- **예상 순서**: ULCC > Hybrid > LCC > Legacy

### H3 (Competition Impact)
- **종속변수**: `HHI`, `Load_Factor`
- **독립변수**: `ULCC_Share`, `Has_ULCC`
- **도구변수**: `IV_Hist_ULCC`

### H4 (COVID Recovery)
- **종속변수**: `Recovery_Months`, `Market_Share_Change`
- **독립변수**: `Carrier_Type`
- **통제변수**: `COVID_Dummy`, Mobility indices

---

## 결과 파일 명명 규칙

### Figures
- `Fig_4_1_H1_Market_Behavior.png`
- `Fig_4_2_H2_Network_Structure.png`
- `Fig_4_3_H3_Competition_Impact.png`
- `Fig_4_4_H4_COVID_Recovery.png`

### Tables
- `Table_4_1_Market_Dynamics.csv`
- `Table_4_2_Network_Metrics.csv`
- `Table_4_3_Regression_Results.csv`
- `Table_4_4_COVID_Analysis.csv`

---

## 주요 설정 변수

### 경로 설정
```python
DATA_PATH = "data/"
OUTPUT_PATH = "report/"
OD_PATH = "data/od/"
T100_PATH = "data/t_100/"
```

### 분석 파라미터
```python
YEARS_ANALYZED = range(2014, 2025)
COVID_PERIOD = (2020, 2021)
PRE_COVID = range(2017, 2020)
POST_COVID = range(2022, 2025)
```

---

## 디버깅

### 일반적인 문제들
1. **데이터 로딩 실패**: `data_loader.py`의 파일 경로 확인
2. **메모리 부족**: 청크 단위 처리 또는 데이터 필터링
3. **가설 검증 실패**: 각 분석 모듈의 `test_hypothesis()` 메서드 확인
4. **시각화 오류**: `visualization_manager.py`의 Figure 사양 확인

### 로그 및 디버깅
- 각 모듈에는 진행상황 출력 포함
- 오류 발생시 모듈별로 독립 실행 가능
- `print()` 문으로 중간 결과 확인 가능

---

## 실행 방법

### 전체 분석 실행
```python
python run_analysis.py
```

### 개별 모듈 테스트
```python
from network_analysis import NetworkAnalyzer
analyzer = NetworkAnalyzer(od_data, carrier_classification)
results = analyzer.analyze_network_structure()
```

### 특정 가설만 실행
```python
from main_analysis import AirlineAnalysisController
controller = AirlineAnalysisController()
h2_results = controller.run_h2_analysis()
```
