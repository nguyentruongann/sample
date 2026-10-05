# Monitoring theo 3 yêu cầu mentor

Gói cập nhật chỉ chứa file thêm/sửa. Chép đúng cây thư mục `demand_forecasting/`
vào dự án hiện tại; giữ nguyên `.env`, dữ liệu, model và những file không có trong gói.
ZIP source mới gửi chưa chứa monitoring, nên bản này tích hợp đầy đủ phần cần thiết.

## 1. Ba phần chính

### Feature

- Ghi tỷ lệ missing/nonfinite của từng feature trước `model.predict` cho cả 6 model.
- Theo dõi số hex không có lịch sử, độ cũ p95/lớn nhất của bucket nguồn và tuổi snapshot.
- Warning mặc định nếu nguồn/snapshot cũ hơn **15 phút**, hoặc một feature thiếu **>5%**.
- Độ cũ nguồn = hiện tại trừ thời điểm đóng bucket mới nhất có sẵn của từng hex.
  Không nhầm lag_7d được thiết kế nhìn lại 7 ngày với dữ liệu nguồn bị trễ 7 ngày.
- Raw hiện tại không có `ingested_at`, nên đây là **freshness**, chưa phải độ trễ
  vận chuyển thực tế giữa event-time và ingestion-time. Khi có data real nên thêm
  timestamp ingestion để đo riêng chỉ số đó.
- Snapshot được ghi cả khi fresh lag thiếu và inference sau đó bị từ chối.
  Lỗi trước khi tạo được feature sẽ được nhận biết qua snapshot thiếu/cũ.
- Dữ liệu fake có hex không hoạt động mỗi ngày: missing không nhất thiết là lỗi;
  điều chỉnh ngưỡng theo hợp đồng dữ liệu, không tự lấp missing bằng 0.

### Performance model

- DAG **`demand_model_monitoring`** chạy mỗi 1 giờ mặc định; có thể đổi thành 2 giờ.
- Mặc định 05:00 chấm dự báo có **forecast_start T trong [03:00,04:00)**.
- H10/H30/H60 đều dùng actual là tổng bucket đúng lưới trong **[T,T+H)**.
- Phải đủ H/10 bucket và đã tới T+H+120 giây mới chấm. Missing actual không là 0.
- Bản ghi phát hành trễ quá 120 giây sau T hoặc sau khi target đã kết thúc bị loại
  khỏi performance, nhưng được đếm riêng. Grace này là dung sai vận hành; muốn
  chấm nghiêm ngặt tại T có thể đặt 0, khi đó job Airflow đến sau T sẽ bị loại.
- Lưu WMAPE, MAE, bias, tổng actual/prediction trên cùng các mẫu được chấm,
  coverage, số mẫu, số mốc forecast, missing/pending/late cho từng model_version.
- Tự đánh giá lại các cửa sổ trong **48 giờ gần nhất** mỗi lượt để nhận actual muộn.
  Re-run là idempotent; hết 48 giờ phải chạy lại với `--as-of` phù hợp nếu cần sửa
  cửa sổ cũ. Actual trong raw đã được cập nhật thì lần chấm tiếp theo sẽ nhận giá trị mới.
- `ready` yêu cầu actual đủ, có >=100 mẫu và tổng actual >=1000. Đây là ngưỡng
  thống kê ban đầu, không phải điều kiện model được phê duyệt production.
- Bảng `monitor_prediction_archive` giữ dự báo lần phát hành đầu, retry không ghi
  đè; bảng `monitor_performance_windows` lưu kết quả cửa sổ. Các bảng được tạo
  thêm, không thay bảng serving hiện tại. Lịch sử bắt đầu từ lần chạy sau nâng cấp.

### Phân phối dự đoán

- So sánh **prediction serving 24 giờ gần nhất** theo `issued_at` với **prediction
  của final model trên đúng TRAIN+VAL đã dùng fit (đã purge)**; tách cả 6 model và version.
- Tham chiếu không lấy actual, không lấy tập test, không lấy 6 ngày serving trước.
- Lưu sidecar `*.distribution.json` cạnh file model: bins, counts, min/max,
  p50/p90/p95, khoảng fit và hash phiên bản model. Cùng clip+rint như serving.
- Chỉ dùng đúng bins/reference cùng version để tính PSI, tỷ lệ ngoài range và
  biểu đồ histogram. Không cần đợi actual để so sánh phân phối.
- Warning khi PSI >0.2 hoặc >5% prediction nằm ngoài min/max reference,
  nếu mỗi phía có >=100 mẫu. Chưa đủ mẫu hiển thị `insufficient_samples`.
- Reference chưa có/không khớp hiển thị `reference_missing`/`reference_invalid`;
  không tự thay bằng dữ liệu gần đây. Nhóm model chưa phục vụ cũng hiển thị chờ mẫu.
- Chênh lệch mùa vụ/giờ/ngày/tập hex cũng gây PSI cao; đây là tín hiệu điều tra,
  không chứng minh accuracy giảm. Reference fit có đặc tính in-sample, không dùng
  nó để báo cáo accuracy hoặc chọn tham số model.

Không tự retrain, không tự hiệu chỉnh prediction. Các phần shock/nhóm lỗi hex–thứ–giờ
và PSI actual 6 ngày cũ không còn nằm trong dashboard mới để giữ đúng phạm vi mentor.
Cảnh báo xem ở Prometheus, chưa cấu hình gửi ra email/Slack.

## 2. Nâng cấp hệ thống đang chạy (Linux/WSL)

Đặt hai file Compose và thư mục monitoring cùng cấp. Trong `.env` hiện tại thêm/cập nhật:

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.monitoring.yml
GRAFANA_ADMIN_PASSWORD=mat_khau_cua_ban
```

PowerShell/CMD Windows dùng `;` thay `:`; WSL/Linux dùng `:`.
Không sửa tên project Compose đã dùng trước đó và không chạy `down -v`.

```bash
docker compose config --services
docker compose build api monitor pipeline airflow-api-server airflow-scheduler airflow-dag-processor
docker compose up -d --no-deps api airflow-api-server airflow-scheduler airflow-dag-processor monitor prometheus grafana
```

Đã khởi tạo database/Airflow rồi thì `--no-deps` tránh chạy lại bootstrap. Với dự án
mới hoàn toàn, hoàn tất `.env`/PostgreSQL theo README và chạy `docker compose up -d --build`.

DAG monitoring mới được cấu hình tự bật khi được tạo. Nếu trước đó đã tồn tại và
đang pause, bật trên Airflow UI hoặc:

```bash
docker compose exec airflow-scheduler airflow dags unpause demand_model_monitoring
```

Dashboard đóng gói trong Python wheel; Docker build sẽ dừng sớm nếu thiếu
`monitoring.html`, tránh lỗi trang chủ 500 do thiếu HTML như bản triển khai trước.
Monitor được mount `model_store` read-only để đọc đúng reference cùng model.
Grafana có dashboard mới thay UID cũ; nó cũng được đặt làm trang Home.

| Trang | Địa chỉ |
|---|---|
| Ba mục monitoring chi tiết + lịch sử cửa sổ | http://127.0.0.1:8001 |
| Grafana: Feature, Performance, Prediction Distribution | http://127.0.0.1:3000 |
| Cảnh báo / kiểm tra scrape | http://127.0.0.1:9090/alerts / http://127.0.0.1:9090/targets |

Prometheus bản này chỉ scrape `demand-monitor`; không cần `demand-api` cho ba mục mentor.
Grafana hiển thị kết quả cửa sổ mới nhất tại thời điểm scrape; xem 8001 để chọn chính
xác khung T và đọc các kết quả được chấm lại. Không coi đường WMAPE trên Grafana là
sai số tức thời mỗi 30 giây.

## 3. Tạo reference cho model đang có — không train lại

Các lần train bằng source mới tự sinh reference trước khi publish manifest.
Với 6 model cũ, chạy **một lần**:

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.prediction_reference
```

Mặc định đọc snapshot feature ban đầu: `/opt/demand_forecasting/data/processed/demand_features.parquet`.
Có thể chỉ rõ file khác:

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.prediction_reference --feature-path /opt/demand_forecasting/data/processed/demand_features.parquet
```

Lệnh dùng metadata train_start/val_start/test_start để phục hồi các hàng fit đúng
purge, kiểm tra số hàng theo `runs/<run_id>/metrics.csv`, thứ tự feature và mapping hex,
rồi dùng **model đã lưu** predict. Không thay trọng số, active.json hoặc model_version.
Reference đã có và hợp lệ sẽ được giữ nguyên.

Phải dùng **snapshot feature gốc của lần train đó**. Các kiểm tra cấu trúc/số hàng
không thể phát hiện mọi thay đổi thủ công về giá trị trong một snapshot cũ. Nếu
file gốc đã mất/bị ghi đè, khôi phục snapshot đúng hoặc train theo source mới để
sinh reference đúng; không dùng prediction serving để lấp chỗ trống.

## 4. OPTION test nhanh — không chờ serving thật

Đây là lệnh chủ động, không nằm trong lịch Airflow và không tự chạy khi `up`.
Serving vẫn real time. Mỗi lần demo dùng schema riêng `<schema>_monitor_demo_<id>`,
đóng băng một bản copy tạm của 6 model đang active, sinh dữ liệu fake và gọi đúng
`predict_and_store` ở từng mốc 10 phút. Không đọc/ghi raw/predictions của schema live,
không chỉnh đồng hồ hệ thống, không đổi model active, không đổi metrics live.

Sau bước tạo reference, thử **3 giờ** trước:

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.monitoring_demo --hours 3
```

Muốn có đủ một ngày serving để thử phân phối:

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.monitoring_demo --hours 24
```

Có thể chọn mốc bắt đầu (phải sau khoảng fit của model, có timezone, đúng đầu giờ):

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.monitoring_demo --hours 3 --start 2026-10-06T08:00:00+07:00
```

Nếu interval=2 thì chọn số giờ chẵn và mốc bắt đầu vào giờ chẵn. CLI hỗ trợ 1–48 giờ.
Không có `sleep` chờ 10 phút; thời gian thực phụ thuộc máy và lượng hex, nên 24 giờ
replay vẫn có thể mất nhiều phút tính toán. Có bước tạo 8 ngày warm-up cho lag_7d.
Không hạ số hex/số model để làm sai pipeline thực tế.

Lệnh in URL dạng:

```text
http://127.0.0.1:8001/demo?schema=public_monitor_demo_ab12cd34
```

**Mở đúng URL được in**, không lấy ví dụ trên. Trang có nhãn TEST NHANH và đồng hồ
ảo. Kết quả demo không đưa vào Grafana/Prometheus của luồng live. Job performance
được gọi với các mốc giờ ảo, vẫn áp dụng horizon, label wait và cùng công thức đánh giá.
Reference thiếu vẫn được báo thiếu, không giả tạo kết quả drift.

Sau forecast cuối, demo tiến thêm thời gian để có actual H60. Vì thế snapshot
feature cuối có thể báo stale trong báo cáo cuối — đó là khoảng thời gian ảo đã
ngừng phát hành forecast để chờ labels, không ảnh hưởng serving thật.
Mỗi demo giữ schema riêng để xem lại; chưa tự xóa dữ liệu demo. Schema được in ra
để quản lý; không xóa schema live. Tài nguyên CPU/DB vẫn dùng chung máy, nên chạy
3 giờ trước, tránh test dài lúc đang train.

## 5. Các option cấu hình

```dotenv
MONITOR_INTERVAL_HOURS=1
MONITOR_DELAY_HOURS=1
MONITOR_RECHECK_HOURS=48
MONITOR_LABEL_WAIT_SECONDS=120
MONITOR_ISSUE_GRACE_SECONDS=120
MONITOR_FEATURE_DELAY_WARNING_SECONDS=900
MONITOR_FEATURE_MISSING_WARNING_RATIO=0.05
MONITOR_SERVING_WINDOW_HOURS=24
MONITOR_MIN_SAMPLES=100
MONITOR_MIN_DEMAND=1000
MONITOR_PSI_WARNING=0.2
MONITOR_RANGE_WARNING_RATIO=0.05
MONITOR_MAX_ROWS=1000000
MONITOR_POLL_SECONDS=60
```

Chỉnh `.env` xong, chạy `up -d --no-deps` cho monitor và ba service Airflow để nhận
biến mới. Scheduler interval chỉ hỗ trợ 1/2 giờ. Delay tính từ **cuối khung T**;
interval=2 và delay=1: lúc 06:00 chấm T trong [03:00,05:00).
Monitor chỉ refresh dashboard/feature/distribution; không tự chấm performance mỗi phút.

Cảnh báo WMAPE 15%, |bias| 10% đang nằm trong `monitoring/alerts.yml`; nếu thay file
rule cần restart Prometheus. Ngưỡng feature và distribution lấy từ env nên đồng
nhất giữa UI và cảnh báo. Các ngưỡng là cấu hình thử, chưa chốt cho dữ liệu thật.

## 6. Kiểm tra và xử lý lỗi

```bash
docker compose logs --tail=80 monitor
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose exec airflow-scheduler airflow dags list
```

Chạy chấm một lượt thủ công cho luồng thật (không tạo data/dự báo mới):

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.performance_monitor
```

Replay lại cửa sổ đánh giá cũ, sử dụng archive/actual đã có:

```bash
docker compose run --rm --no-deps pipeline python -m demand_forecasting.performance_monitor --as-of 2026-10-05T05:00:00+07:00
```

Test tự động:

```bash
python -m pytest tests -q
```

Test tích hợp PostgreSQL cần `MONITOR_TEST_DATABASE_URL` trỏ tới database kiểm thử.
Test tạo schema ngẫu nhiên riêng rồi xóa đúng schema đó. Không dùng DB vận hành.

Giới hạn: query actual tối đa 120 giây/cửa sổ; serving tối đa MONITOR_MAX_ROWS.
Vượt giới hạn báo lỗi, không âm thầm lấy mẫu. Recheck chạy tối đa 48 cửa sổ mặc định;
quy mô lớn cần aggregation incremental/partition. Archive chưa tự dọn retention.
Monitor/API một worker theo Dockerfile; không bật nhiều collector worker.

Tài liệu tham khảo cấu hình:
https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html
https://setuptools.pypa.io/en/stable/userguide/datafiles.html
