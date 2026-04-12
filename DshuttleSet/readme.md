# Badminton Analytics Pipeline (DshuttleSet)

這是一個羽球賽事數據特徵工程與視覺化分析專案。專案經過職責分離重構，以保持流程的清晰與模組化。

## Structure

```text
DshuttleSet/
│
├── src/
│   ├── data_pipeline/         # 資料前處理相關 (包含刪除資料、新增特徵)
│   │   ├── __init__.py
│   │   ├── clean.py           # 負責初步清理、刪除無效/異常資料
│   │   └── feature_engineering.py # 負責新增空間與戰術特徵、轉換欄位以及修正return height
│   │
│   ├── analysis/              # 負責計算與進階分析邏輯
│   │   ├── __init__.py
│   │   ├── general_analyze.py # 一般分析邏輯 (球種分佈、擊球區間)
│   │   ├── defensive_failure.py # 防守失敗率熱力圖分析
│   │   |── mid_cover.py       # 中場覆蓋與反應率分析
│   │   └── matches_analyze.py   # 以Level 來區分match之分析
│   │
│   ├── visualization/         # 負責靜態繪圖、產生視覺化輔助圖
│   │   ├── __init__.py
│   │   └── draw_mid_position.py # 負責繪製標準羽球場與中位圖表
│   │
│   └── utils/                 # 共用工具程式碼
│       ├── __init__.py
│       └── util.py            # 常用的共用函式，如讀寫檔案等
│
├── data/                  # 放置原始 CSV 資料
├── img/                   # 視覺化匯出的靜態資源與分析結果圖片檔
├── requirements.txt       # 專案相依性套件
└── main.ipynb             # 專案主進入點，負責資料流與呼叫各模組
