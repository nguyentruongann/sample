# Cập nhật pipeline theo final.ipynb

Bản này dùng **6 model gốc**, không bật hiệu chỉnh tuyến tính/residual. Đây là
chuyển logic notebook thành pipeline train và inference, chưa phải chứng nhận
chất lượng triển khai thực tế.

## 1. Cấu hình đã chốt

Nguồn: output `FINAL_FIT_SUMMARY` trong `final.ipynb` do bạn cung cấp.

| Mode | Horizon | Feature | Variant | num_leaves | n_estimators | learning_rate |
|---|---:|---:|---|---:|---:|---:|
| Car | 10 | 25 | fresh_10_20 | 63 | 633 | 0.05 |
| Car | 30 | 33 | fresh_plus_neighbor | 130 | 763 | 0.06 |
| Car | 60 | 33 | fresh_plus_neighbor | 130 | 799 | 0.06 |
| Motorcycle | 10 | 25 | fresh_10_20 | 63 | 762 | 0.05 |
| Motorcycle | 30 | 33 | fresh_plus_neighbor | 63 | 795 | 0.06 |
| Motorcycle | 60 | 33 | fresh_plus_neighbor | 130 | 792 | 0.06 |

Cấu hình chung: LightGBM CPU, objective Poisson, metric L1,
min_child_samples=100, subsample=0.90, subsample_freq=1,
colsample_bytree=0.90, reg_lambda=2.0, seed=20260916. Không giới hạn max_depth.
CPU thread mặc định 4 để dùng chung máy với inference; notebook dùng -1.

`src/demand_forecasting/notebook_contract.py` lưu cấu hình cố định, thứ tự
feature bổ sung và version schema. Target Hh là tổng demand **[T,T+H)**;
không có bước dự đoán H10 rồi đưa kết quả đó vào model H30/H60.

## 2. Luồng sau khi sửa

1. Đọc raw từ PostgreSQL; benchmark lịch sử chỉ đọc trước 11/09/2026.
2. Tạo 21 feature nền và lọc hàng thiếu lag T−30 như notebook.
3. Tạo fresh/neighbor trên lịch sử đã lọc, **trước** khi lọc target riêng từng horizon.
4. Tạo target và naive features ở đúng timestamp, không shift theo vị trí hàng.
5. Chia train/VAL/test; purge các target vượt ranh giới train→VAL và VAL→test.
6. Fit model TRAIN để báo cáo VAL; refit model cuối trên TRAIN+VAL đã purge.
7. Clip và làm tròn prediction trước khi tính metric TEST; lưu 6 artifact và metadata.
8. Flow rolling giữ nguyên cơ chế kích hoạt nguyên bộ model của dự án cũ.
9. Inference đọc artifact và dựng cùng feature, đúng mã hex, đúng gốc trend_day,
   chỉ dùng dữ liệu trước T; clip/rint giống đánh giá test. Không train khi gọi predict.

Neighbor là trung bình demand của các H3 kề một vòng, cùng mode, loại chính hex
đang dự đoán; bỏ qua hàng xóm thiếu quan sát, không biến missing thành demand=0.
Hex không có hàng xóm khả dụng nhận NaN để LightGBM xử lý. Với bộ hex cực nhỏ
không có cạnh nào, pipeline trả NaN thay vì lỗi như helper notebook; trên bộ hex
có cạnh, test đối chiếu xác nhận các giá trị khớp notebook.

`hex_code` H10 giữ mã từ danh sách hex toàn bộ dữ liệu, truyền DataFrame với
categorical_feature. H30/H60 factorize lại theo mode/horizon sau lọc target và
truyền numeric float32 như notebook. Artifact lưu `hex_code_mapping` để online
không đánh số lại theo dữ liệu vừa tải. Ngày lễ được tính lại từ lịch Việt Nam.

## 3. Hai chế độ train

| Chế độ | Cách chọn số lá/cây | Số lần fit cho 6 model |
|---|---|---:|
| `locked` — mặc định | Dùng đúng bảng đã chốt ở trên; không early stopping/tuning lại | 6 fit TRAIN báo cáo VAL + 6 refit = 12 |
| `notebook_search` | Thử lá H10: 130→31→63; H30/H60: 63→31→130; tối đa 800 cây, early stopping 50 vòng | 18 search fit + 6 refit = 24 |

Search chọn theo WMAPE float không âm trên **toàn VAL**; hòa điểm giữ trial
xuất hiện trước. `SEARCH_ESTIMATORS=800` áp dụng cho search, không ghi đè cây
đã chốt trong `locked`. `SEARCH_TRAIN_ROWS`/`SEARCH_VAL_ROWS` cũ không còn làm
pipeline lấy mẫu. `MAX_TRAIN_ROWS` phải để trống; nếu khác, pipeline báo lỗi
thay vì âm thầm train sai dữ liệu notebook.

### Phân biệt lịch sử và rolling

- `train` không có `--rolling`: train toàn lịch sử trước VAL, VAL 30 ngày,
  test từ 15/08/2026 09:40 +07 đến hết lịch sử gốc trước 11/09. Không cắt train
  còn 90 ngày. Chỉ tạo báo cáo/artifact candidate, không thay active.json.
- Bootstrap / DAG hàng tuần / `train --rolling`: giữ lịch vận hành cũ, tối đa
  90 ngày train, 30 ngày VAL, 30 ngày test mới nhất. Feature và cách fit giống
  notebook, nhưng dữ liệu/cửa sổ khác, **không kỳ vọng WMAPE giống bảng notebook**.
- `locked` trên cửa sổ mới là tái sử dụng hyperparameter đã chốt, không có nghĩa
  đã tìm được số cây tối ưu trên cửa sổ mới.

## 4. Cách áp dụng source thay đổi

ZIP này chỉ chứa file sửa/thêm, có thư mục gốc `demand_forecasting/`.
Giải nén, chép đè các file theo đúng đường dẫn vào project hiện có. Không thay
`.env` chứa tài khoản PostgreSQL của bạn; chỉ sửa/thêm những dòng dưới đây:

```dotenv
TRAINING_STRATEGY=locked
FRESH_BUCKETS_CLOSED_AT_T=true
MAX_TRAIN_ROWS=
SEARCH_ESTIMATORS=800
SEED=20260916
LOCAL_TIMEZONE=Asia/Ho_Chi_Minh
TEST_START=2026-08-15T09:40:00+07:00
VALIDATION_DAYS=30
MODEL_NAMES=lightgbm
HORIZONS=10,30,60
```

`FRESH_BUCKETS_CLOSED_AT_T=true` là điều kiện dữ liệu phải đáp ứng: bucket
T−10/T−20 đã đóng và ingest trước lúc forecast. Source từ chối false vì feature
đã chốt cần dữ liệu này. Cờ này không tự kiểm chứng độ trễ ingest thực tế;
flow fake hiện tại backfill bucket đóng trước inference. Khi thay nguồn thật,
cần xác nhận điều kiện bằng timestamp ingest/watermark của hệ thống thật.

### Chạy lại flow Docker hiện có

Tại thư mục chứa `docker-compose.yml`, sau khi cập nhật code và `.env`:

```powershell
# Dừng container cũ khi chuyển schema; giữ nguyên named volumes và dữ liệu.
docker compose down
# Rebuild các image chứa source mới rồi khởi động flow.
docker compose up -d --build
docker compose logs -f bootstrap
```

Không thêm `-v` vào lệnh down. Bootstrap nhận schema cũ 21/24 feature và train
lại theo schema mới 25/33 feature. Model run cũ vẫn còn trong volume; chỉ
active.json được thay khi run mới đạt điều kiện flow. Restart sau đó sẽ bỏ qua
train nếu active model hợp lệ. Đây là thay đổi source; không kèm model mới
đã train trên toàn bộ dữ liệu của bạn.

Flow cũ tự kích hoạt model rolling khi đủ 6 artifact, VAL/TEST hữu hạn và mọi
TEST WMAPE ≤30%. Bản sửa giữ ngưỡng này cho **demo dữ liệu giả**, tính trên
prediction integer. Ngưỡng này không thay thế kết luận notebook về hạn chế
bám spike và chưa phải gate production.

### Đối chiếu notebook sau khi DB đã có lịch sử

```powershell
# Build cả service pipeline thuộc profile tools.
docker compose --profile tools build pipeline
# Dùng đúng cấu hình số lá/cây đã chốt trên cửa sổ lịch sử.
docker compose --profile tools run --rm pipeline python -u -m demand_forecasting.pipeline train --strategy locked
# Hoặc chạy lại toàn bộ quy trình search giống notebook.
docker compose --profile tools run --rm pipeline python -u -m demand_forecasting.pipeline train --strategy notebook_search
# Xuất diagnostics của lần benchmark vừa chạy.
docker compose --profile tools run --rm pipeline python -u -m demand_forecasting.pipeline evaluate
```

Chọn một trong hai lệnh train theo mục đích; không cần chạy cả hai để deploy
flow demo. Để retrain rolling thủ công, thêm `--rolling` vào lệnh train.

## 5. Báo cáo và artifact

- Benchmark: `outputs/notebook_benchmark_summary.csv` và
  `outputs/notebook_benchmark_search_trials.csv`.
- Rolling: `outputs/weekly_retrain_summary.csv` và `outputs/model_search_trials.csv`.
- Từng run: `models/runs/<run_id>/manifest.json`, `metrics.csv`,
  `model_search_trials.csv`, `validation_<mode>_h<horizon>_lightgbm.parquet`,
  6 file `.joblib`. H10 được lưu đầy đủ như H30/H60.
- Prediction TEST: `outputs/predictions_*.parquet` cho benchmark hoặc
  `outputs/weekly/predictions_*.parquet` cho rolling.
- `WMAPE_%` là metric integer dùng so sánh với bảng test notebook;
  `WMAPE_float_%` chỉ là số tham khảo. `VAL_WMAPE_%` là float để chọn model;
  `VAL_WMAPE_integer_%` là kết quả sau làm tròn.
- Artifact khai báo `feature_schema=final_fresh_neighbor_v1`,
  `postprocessing=clip_rint_only`. Model schema cũ bị từ chối khi serving.

## 6. File sửa/thêm

| File | Vai trò |
|---|---|
| `src/demand_forecasting/notebook_contract.py` (mới) | Cấu hình 6 model và feature schema |
| `src/demand_forecasting/config.py` | Chế độ locked/search và điều kiện fresh bucket |
| `src/demand_forecasting/feature_engineering.py` | Fresh, neighbor, target và hex-code đúng notebook |
| `src/demand_forecasting/data_split.py` | Benchmark dùng toàn bộ lịch sử train |
| `src/demand_forecasting/train.py` | Fit theo cấu hình đã chốt/search, refit, metric integer, metadata |
| `src/demand_forecasting/predict.py` | Feature online và hex mapping tương thích artifact |
| `src/demand_forecasting/pipeline.py` | CLI --strategy và nhận diện model schema mới |
| `dags/retrain_dag.py` | Báo cáo trạng thái train theo chiến lược thực tế |
| `.env.example`, `docker-compose.yml` | Truyền cấu hình mới cho app/Airflow |
| `tests/test_final_notebook_contract.py` (mới) | Parity với notebook và train locked thật |
| `tests/fixtures/final_notebook_feature_cells.json` (mới) | Code gốc các cell 8/9/11 để test, không chạy trong production |
| `tests/test_direct_lightgbm.py`, `tests/test_model_promotion.py` | Cập nhật test search và mock metric integer |
| `README.md`, `VALIDATION.md`, `NOTEBOOK_MIGRATION.md` | Hướng dẫn và phạm vi kiểm chứng |

Xem `VALIDATION.md` cho kết quả test và các giới hạn chưa kiểm tra.
