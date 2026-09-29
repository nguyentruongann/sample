# Demand Forecasting — Docker + Airflow + Prediction API

Hệ thống mô phỏng demand theo H3 tại TP.HCM, lưu dữ liệu vào PostgreSQL chạy
**ngoài Docker**, train
**chỉ LightGBM CPU** theo `notebooks/source_main.ipynb` (bản notebook gốc được
kèm trong ZIP) và tạo dự đoán H10/H30/H60 mỗi
10 phút bằng Apache Airflow. Notebook là tham chiếu cho dữ liệu fake và loại
model LightGBM; cấu hình cây được tìm trên validation. Docker Compose và Airflow
tự vận hành, không cần train tay.

H10 dự đoán demand bucket `[T,T+10)`. H30/H60 dự đoán **tổng** lần lượt ba/sáu
bucket kể từ T. Ba model dùng feature lịch sử tại T và được train độc lập theo
Car/Motorcycle; H30/H60 **không nhận output của H10**.

## Luồng chạy

```text
Airflow (*/10 phút)
  ├─ bù bucket fake còn thiếu đến hiện tại
  ├─ UPSERT vào PostgreSQL demand_db
  ├─ load toàn bộ model LightGBM đã duyệt từ một phiên bản trong model_store
  ├─ dự đoán H10/H30/H60 cho bucket kế tiếp
  └─ UPSERT vào public.demand_predictions

FastAPI
  └─ đọc prediction mới nhất từ PostgreSQL và trả JSON
```

Airflow có database metadata riêng `airflow`; dữ liệu nghiệp vụ nằm trong
`demand_db`. Cả hai database đều nằm trên PostgreSQL của máy host hoặc server
riêng. Docker Compose **không chạy container PostgreSQL** và **không chứa dữ
liệu database trong image/volume Docker**. Thư mục dữ liệu thật do dịch vụ
PostgreSQL ngoài Docker quản lý; kiểm tra bằng `SHOW data_directory;` trong
pgAdmin hoặc `psql`. Các Docker volume còn lại chỉ chứa file Parquet, model,
output và log Airflow.

## PostgreSQL ngoài Docker và các container ứng dụng

| Thành phần | Vai trò | Địa chỉ mặc định |
|---|---|---|
| PostgreSQL trên máy host | `demand_db` và `airflow` | `localhost:5432` trên host; `host.docker.internal:5432` từ container |
| `airflow-api-server` | Airflow UI/API | `http://localhost:8080` |
| `airflow-scheduler` | Lập lịch và chạy LocalExecutor | Chỉ nội bộ Docker |
| `airflow-dag-processor` | Parse DAG | Chỉ nội bộ Docker |
| `api` | Prediction API và dashboard bản đồ H3 | `http://localhost:8000/dashboard` |
| `bootstrap` | Tạo DB nếu thiếu, tự sinh fake data, train và bù lịch sử | Tự chạy trước API/Airflow |
| `airflow-activate` | Tự bật DAG 10 phút và retrain | Tự chạy sau khi Airflow sẵn sàng |
| `pipeline` | CLI generate/train/predict | Chỉ chạy khi gọi lệnh |

## 1. Chuẩn bị PostgreSQL trên host và `.env`

Cài/chạy PostgreSQL **trên máy host** (PostgreSQL 16 nếu chuyển từ stack cũ),
hoặc dùng server PostgreSQL riêng. Cần hai database độc lập: `demand_db` và
`airflow`. `init-runtime` tự tạo role và database còn thiếu khi `.env` có
`POSTGRES_ADMIN_URL` trỏ tới tài khoản PostgreSQL đã tồn tại, có quyền tạo
role/database. Không cần vào pgAdmin để tạo chúng bằng tay. Tương đương SQL:

```sql
CREATE ROLE demand_user LOGIN PASSWORD 'your_demand_password';
CREATE DATABASE demand_db OWNER demand_user;
CREATE ROLE airflow LOGIN PASSWORD 'your_airflow_db_password';
CREATE DATABASE airflow OWNER airflow;
```

Nếu role/database đã tồn tại, giữ nguyên và chỉ cấu hình thông tin đăng nhập
phù hợp. PostgreSQL cần nhận kết nối TCP từ Docker: đặt `listen_addresses`
bao gồm giao diện mà `host.docker.internal` trỏ tới, cho phép **đúng subnet
Docker** trong `pg_hba.conf` với `scram-sha-256`, và giới hạn truy cập bằng
firewall. Trên Docker Desktop, hostname từ container mặc định là
`host.docker.internal`; trên Linux, Compose đã khai báo `host-gateway`. Nếu
PostgreSQL nằm trên server khác, thay hostname trong **hai URL** của `.env`
bằng DNS/IP của server đó và cho phép subnet Docker kết nối tới server.

Lần cài mới, copy `.env.example` thành `.env` và điền password/secret.
Khi nâng cấp, giữ nguyên `.env` đang dùng. File ZIP không kèm thông tin đăng nhập. Đổi tối thiểu các biến sau:

```dotenv
DATABASE_URL=postgresql://demand_user:...@host.docker.internal:5432/demand_db
AIRFLOW_METADATA_URL=postgresql+psycopg2://airflow:...@host.docker.internal:5432/airflow
POSTGRES_ADMIN_URL=postgresql://postgres:...@host.docker.internal:5432/postgres
AIRFLOW_ADMIN_PASSWORD=...
AIRFLOW_FERNET_KEY=...
AIRFLOW_API_JWT_SECRET=...
```

`POSTGRES_ADMIN_URL` có thể để trống nếu cả hai role/database đã tồn tại và URL
trên dùng đúng password của chúng. Tài khoản `postgres` trong ví dụ là tài
khoản PostgreSQL, không phải tài khoản đăng nhập pgAdmin. Chương trình không
tự thay password role cũ; tài khoản admin chỉ đi vào container khởi tạo.

`DATABASE_URL` được app/CLI/API dùng trong **container**;
`AIRFLOW_METADATA_URL` chỉ dành cho Airflow, trỏ đến database khác trên cùng
server. Các URL phải dùng password đúng với các role đã tạo. Nếu password chứa
`@`, `:`, `/`, `#`, hãy URL-encode phần password trong URL (ví dụ `@` → `%40`).
Khi chạy Python trực tiếp trên host, đổi riêng hostname thành `localhost`
bằng biến môi trường trong terminal như hướng dẫn cuối README.

Có thể tạo Fernet key mới bằng Docker:

```powershell
docker run --rm apache/airflow:3.3.2-python3.12 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy kết quả vào `AIRFLOW_FERNET_KEY`.
Tạo JWT secret bằng `docker run --rm apache/airflow:3.3.2-python3.12 python -c "import secrets; print(secrets.token_hex(32))"` rồi copy vào `AIRFLOW_API_JWT_SECRET`.

## 2. Build và khởi động toàn bộ Docker stack

Trước khi chuyển từ bản cũ có PostgreSQL trong Docker, xem mục **Chuyển dữ
liệu cũ** bên dưới. Chạy trong PowerShell tại thư mục project:

```powershell
docker compose down
docker compose up -d --build
docker compose ps -a
docker compose logs -f bootstrap
```

Lần đầu `docker compose up` có thể đợi tới khi sinh dữ liệu và train xong;
muốn theo dõi tiến độ trong lúc đợi, mở PowerShell khác tại thư mục dự án và
chạy `docker compose logs -f bootstrap`.

Compose tự chạy `bootstrap`: tạo database/role còn thiếu, tạo bảng, sinh và
nạp fake history **chỉ khi bảng chưa có dòng**, bù tới bucket vừa đóng,
train/evaluate LightGBM khi chưa có model phù hợp với phiên bản fake hiện tại, bù
tiếp bucket phát sinh lúc train, rồi dự đoán. Lần nâng cấp này tự tạo lại
**phần fake live từ 11/09/2026** theo phân phối và độ mượt thời gian của notebook; lịch sử notebook
trước mốc này được giữ nguyên. Việc làm mới chỉ chạy một lần theo phiên bản
generator; các lần khởi động sau không lặp lại hoặc train lại nếu model hợp lệ.
Sau bootstrap, `airflow-init` migrate metadata, `airflow-activate` tự bật DAG
fake inference 10 phút và retrain hàng tuần (kể cả DAG đã từng bị pause).

Nếu cần kiểm tra riêng lệnh từ log, lệnh này **chỉ tạo database/role/bảng**:

```powershell
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline init-runtime
```

Thiếu `demand_db` và chưa điền `POSTGRES_ADMIN_URL` sẽ báo cụ thể ngay, không
đợi 30 lần rồi báo “not ready”. Nếu không kết nối được host, kiểm tra
`DATABASE_URL`, PostgreSQL, `listen_addresses`, `pg_hba.conf` và firewall.

`airflow-init` migrate metadata database, tạo tài khoản admin, Pool `ml_cpu`
cho train và Pool `db_write` để tuần tự hóa ingest/dự báo live. DAG inference
được bật sau khi bootstrap đã có model.

Kiểm tra log:

```powershell
docker compose logs --tail 100 airflow-api-server
docker compose logs --tail 100 airflow-scheduler
docker compose logs --tail 100 api
```

Log PostgreSQL được quản lý bởi dịch vụ PostgreSQL **trên host**, không có
`docker compose logs postgres`.

Các địa chỉ trên máy host:

- Airflow: `http://localhost:8080`
- Dashboard H3: `http://localhost:8000/dashboard`
- Prediction API: `http://localhost:8000`
- Swagger API: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432` trên host (hoặc địa chỉ server bạn cấu hình)

Đăng nhập Airflow bằng `AIRFLOW_ADMIN_USERNAME` và
`AIRFLOW_ADMIN_PASSWORD` trong `.env`.

## 3. Bootstrap dữ liệu và train model lần đầu bằng Docker

`docker compose up -d --build` ở mục 2 đã làm việc này tự động. Lệnh sau
chỉ dùng nếu muốn chạy lại một lần riêng để xem log; nó kiểm tra dữ liệu/model
trước, **không xóa bảng đang có dữ liệu**:

```powershell
docker compose --profile tools run --rm pipeline python -u -m demand_forecasting.pipeline all --models lightgbm --horizons 10,30,60
```

Notebook có bảng H10 chọn riêng theo validation: Car XGBoost 15,38% so với
LightGBM 15,46%, Motorcycle LightGBM 14,91%. Tuy nhiên bảng **tổng** chọn
LightGBM (overall validation 15,12%), và cell ngay sau bảng ghi chọn LightGBM
để dễ quản lý; phần phân tích sâu cũng dùng LightGBM direct cho H10/H30/H60.
Luồng vận hành này theo quyết định cuối của notebook: sáu model LightGBM.

Bootstrap và weekly retrain dùng cùng một cách chia theo thời gian cho cả sáu
cặp Car/Motorcycle × H10/H30/H60: **30 ngày mới nhất làm test**, 30 ngày trước
đó làm validation, và tối đa 90 ngày trước validation để train. Nếu chưa đủ
90 ngày lịch sử, dùng tất cả các ngày sẵn có trước validation; source notebook
bắt đầu 01/05/2026 nên khi chạy cuối tháng 9, train mới gần ba tháng.
Target vượt qua ranh giới tập bị loại. **Báo cáo rolling phục vụ dự báo** có
test 30 ngày gần nhất. Benchmark đối chiếu notebook là báo cáo riêng: train
trước validation 30 ngày, test từ `TEST_START=2026-08-15T09:40:00+07:00`
đến hết lịch sử gốc `2026-09-11T00:00:00+07:00`. Dữ liệu sinh sau 11/9
không vào benchmark này, và benchmark không thay `models/active.json`.

Source dùng LightGBM Poisson CPU và feature từ notebook, **không lấy sẵn số lá
hoặc số cây đã thắng của notebook**. Với từng mode × horizon, nó thử số lá
`31, 63, 130` như notebook, không áp thêm giới hạn `max_depth`, early stopping 50 vòng
trên validation, rồi chọn WMAPE validation thấp nhất. Search mặc định dùng
tối đa 300.000 train và 100.000 validation rows để tiết kiệm CPU; điểm chọn
cây `VAL_WMAPE_%` được tính trên **toàn bộ validation**. Model cuối fit lại
trên toàn bộ train+validation bằng số cây đã chọn; test chỉ dùng để đo WMAPE.
H10 dùng `hex_code` categorical; H30/H60 dùng ma trận float32 với `hex_code`
numeric, đúng cách truyền dữ liệu ở notebook.
Có 3 lần fit tìm cây + 1 lần fit cuối cho mỗi model (24 lần fit cho cả sáu).
Trước lúc fit, pipeline so WMAPE của baseline lag 30 phút giữa tuần lịch sử
notebook và tuần fake live gần nhất. Nếu live lệch quá lớn, nó dừng trước khi
tốn thời gian train. Log in validation/test WMAPE và tách test thành trước/sau 11/09 để phát hiện
generator live bị lệch. Trial ở `outputs/model_search_trials.csv`,
bootstrap ở `outputs/weekly_retrain_summary.csv`, và diagnostics ở
`outputs/diagnostics/`. Vì có tìm cây, lần train đầu sẽ lâu hơn bản trước.

Chỉ khi đủ sáu model, VAL/TEST WMAPE hữu hạn và TEST WMAPE từng model không
vượt 30% thì `models/active.json` mới được thay nguyên bộ. Nếu thiếu horizon
hoặc WMAPE vượt ngưỡng, phiên bản cũ vẫn giữ và cảnh báo ghi tại
`outputs/training_alerts.json`. Model của phiên bản generator cũ cũng không
được phục vụ bằng dữ liệu fake mới; bootstrap sẽ báo lỗi thay vì phục vụ model
không đạt.

Dataset đầy đủ hơn 5 triệu dòng nên bước đầu có thể tốn RAM và thời gian. Muốn
test CPU nhanh, đặt trong `.env`:

```dotenv
MAX_TRAIN_ROWS=300000
```

Sau khi train, kiểm tra phiên bản model trong Docker volume:

```powershell
docker compose --profile tools run --rm pipeline sh -lc "ls -lh /opt/demand_forecasting/models /opt/demand_forecasting/models/runs"
```

Có thể dùng DAG `demand_bootstrap` để train/evaluate lại theo ý muốn;
task sinh dữ liệu của DAG cũng kiểm tra bảng trước khi nạp.

## 4. Dùng model đã train để dự đoán bằng Docker

Notebook tạo lịch sử giả đến 11/09/2026. Generator live dùng **phân phối mark
đã hiệu chỉnh** (mean, quantile, tail), profile giờ/ngày/khu vực H3 và lịch ô
H3 hoạt động riêng cho Car/Motorcycle của notebook. Khoảng 150 ô Car và 133 ô
Motorcycle có quan sát mỗi ngày (thay đổi theo ngày); các bucket 10 phút trong
ngày hoạt động là liên tục. Nhiễu theo từng H3 được làm mượt theo cửa sổ
80 phút như notebook, để nhu cầu hai bucket gần nhau có tương quan tương tự
lịch sử. Zero thưa, chủ yếu ban đêm, không liên tiếp trên
cùng H3. Mẫu lịch quan sát gốc được lặp theo chu kỳ 133 ngày để tiếp nối sau
11/09/2026. Đây là phần nối tiếp theo quy luật thống kê, không giữ nguyên mọi
dòng hay toàn bộ thống kê hữu hạn của notebook. Model vẫn phục vụ dự báo cho
217 ô H3/mode, kể cả ô không có quan sát ở bucket vừa qua.
Bootstrap bù tới bucket vừa đóng, DAG tiếp tục sau khi máy từng dừng. Lệnh
dưới chỉ cần khi muốn tự khôi phục dữ liệu ngoài lịch Airflow:

```powershell
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline backfill-live
```

Khi cần kiểm tra riêng một chu kỳ ngoài lịch Airflow:

```powershell
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline ingest-live
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline predict --models lightgbm --horizons 10,30,60
```

Lệnh `ingest-live` sinh bucket 10 phút vừa đóng; chạy lại không tạo duplicate
hoặc ghi đè dữ liệu đã có. Lệnh `predict` lấy timestamp mới nhất trong PostgreSQL,
dự đoán bucket kế tiếp rồi lưu vào `public.demand_predictions`.

Kiểm tra trực tiếp bằng `psql` cài trên host (hoặc chạy câu SQL tương tự trong
pgAdmin):

```powershell
psql -h localhost -p 5432 -U demand_user -d demand_db -c "SELECT forecast_start_utc, travel_mode, horizon_minutes, COUNT(*) FROM public.demand_predictions GROUP BY 1,2,3 ORDER BY 1 DESC,2,3 LIMIT 20;"
```

Gọi API từ host:

```powershell
curl.exe http://localhost:8000/health
curl.exe "http://localhost:8000/v1/predictions/latest?travel_mode=Car&horizon_minutes=10&limit=20"
```

PowerShell thuần:

```powershell
Invoke-RestMethod "http://localhost:8000/v1/predictions/latest?travel_mode=Motorcycle&horizon_minutes=30&limit=20"
```

### Dashboard bản đồ H3 để trình bày

Mở **http://localhost:8000/dashboard** sau khi bootstrap hoàn thành và
`demand_live_inference` đã lưu dự báo. Nếu đang chạy source cũ, ở thư mục
project chạy:

```powershell
docker compose up -d --build api
```

Dashboard đọc trực tiếp các bảng hiện có, không cần tạo bảng hay chạy thêm
service. Có thể chuyển Car/Motorcycle, chọn H10/H30/H60, xem từng ô H3, chuyển
giữa bản đồ dự báo và bucket đã quan sát, xem top 5 điểm nóng, lịch sử 2 giờ,
và phát lại 18 mốc dự báo gần nhất. Ở chế độ **Mới nhất**, trang kiểm tra
kết quả mới mỗi 30 giây; nếu kết quả chưa được cập nhật trên 25 phút, nhãn
trạng thái sẽ báo dự báo cũ. Chọn một mốc lịch sử sẽ tạm dừng việc tự nhảy
sang dự báo mới.

**Cách đọc:** H10 dự báo tổng nhu cầu 10 phút từ thời điểm bắt đầu;
H30/H60 lần lượt là **tổng** trong 30/60 phút từ cùng thời điểm, không phải
nhu cầu của riêng bucket ở phút thứ 30/60. Nút **Đã quan sát** hiển thị bucket
10 phút mới nhất với thời điểm riêng; số này không được dùng để so trực tiếp
với tổng H30/H60. Dữ liệu trên dashboard là **giả lập**; các ô H3 có tọa độ
thật nhưng vùng H3 hiện tại được sinh quanh tâm thành phố, không phải ranh
giới hành chính của TP.HCM. Trang tự vẽ các ô nên vẫn hiển thị nếu không có
dịch vụ bản đồ nền bên ngoài.

API phục vụ riêng cho dashboard:

```powershell
curl.exe "http://localhost:8000/v1/dashboard/snapshot?travel_mode=Car&horizon_minutes=10"
```

## 5. Airflow chạy mỗi 10 phút

`airflow-activate` tự bật DAG sau khi model sẵn sàng. Các lệnh sau chỉ để
kiểm tra hoặc kích hoạt lại theo ý muốn:

```powershell
docker compose exec airflow-scheduler airflow dags unpause demand_live_inference
docker compose exec airflow-scheduler airflow dags trigger demand_live_inference
```

DAG `demand_live_inference` có schedule:

```text
*/10 * * * *
```

Ví dụ run lúc `10:10` sẽ:

1. Sinh actual cho bucket `[10:00, 10:10)`.
2. Lưu bucket vào PostgreSQL.
3. Dự đoán bắt đầu tại `10:10` cho H10/H30/H60.
4. Lưu prediction để API phục vụ.

Khi Airflow chạy muộn hoặc retry, task dự báo kiểm tra và bù bucket còn thiếu
tới thời điểm chạy thật, rồi đọc timestamp mới nhất từ PostgreSQL. Nó không
dùng lại mốc bucket cũ lưu trong XCom của task ingest. Hai task live dùng
Pool `db_write` để không ghi chồng nhau.

Theo dõi:

```powershell
docker compose logs -f airflow-scheduler airflow-dag-processor
docker compose exec airflow-scheduler airflow dags list-runs -d demand_live_inference
```

## 6. Retrain hàng tuần

DAG `demand_weekly_retrain` chạy lúc 02:00 thứ Hai theo
`Asia/Ho_Chi_Minh` và cũng được bật tự động. Nếu bạn đã pause thủ công,
có thể bật lại bằng:

```powershell
docker compose exec airflow-scheduler airflow dags unpause demand_weekly_retrain
```

Task train dùng Pool `ml_cpu`, ingest/dự báo dùng Pool `db_write`; train mặc
định 4 luồng CPU (`TRAIN_N_JOBS`) để DAG 10 phút vẫn chạy trong khi train.
Weekly retrain bù dữ liệu tới hiện tại, tìm cấu hình cây bằng validation
30 ngày, đo WMAPE test 30 ngày và fit trên tối đa 90 ngày train cộng
validation trước khi thay `active.json`. Kết quả ở
`outputs/weekly_retrain_summary.csv`, trial ở `outputs/model_search_trials.csv`,
dự báo test ở `outputs/weekly/`. Model phục vụ không fit trên 30 ngày test;
mốc test 15/08 của lệnh backtest lịch sử không thay đổi.

## 7. Chạy từng bước trong container

Các lệnh này dành cho debug. `load-postgres` **thay toàn bộ bảng raw** nên
không dùng để khởi chạy thông thường; `bootstrap` tự seed an toàn khi bảng rỗng.

```powershell
# Sinh file lịch sử đầy đủ
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline generate

# COPY Parquet vào PostgreSQL
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline load-postgres --parquet /opt/demand_forecasting/data/raw/xanhsm_hcm_smooth_10min_bucket_v9.parquet

# Benchmark notebook riêng: test chỉ đến 11/09, không thay model đang phục vụ
docker compose --profile tools run --rm pipeline python -u -m demand_forecasting.pipeline train --models lightgbm --horizons 10,30,60

# Retrain production với train/validation/test rolling và WMAPE
docker compose --profile tools run --rm pipeline python -u -m demand_forecasting.pipeline train --rolling --models lightgbm --horizons 10,30,60

# Xuất diagnostics cho model bootstrap/weekly
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline evaluate --rolling

# Xem toàn bộ CLI
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline --help
```

## 8. Test source bằng Docker

Image production không chứa pytest. Chạy test bằng container Python tạm:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace python:3.12-slim sh -lc "pip install -r requirements.txt && pip install -e . --no-deps && pytest"
```

## 9. Chuyển dữ liệu cũ từ hai container PostgreSQL

Nếu trước đây bạn đã chạy bản Compose cũ, **không khởi tạo database trống đè
lên dữ liệu cũ**. Trong khi hai container `demand-postgres` và
`demand-airflow-db` còn chạy với volume cũ, sao lưu hai database ra file trên
host. Backup nằm **ngoài thư mục project**; `docker cp` giữ nguyên bytes trên
PowerShell:

```powershell
$backupDir = Join-Path $HOME 'demand-forecasting-backups'
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
docker exec demand-postgres pg_dump -U demand_user -d demand_db -Fc -f /tmp/demand_db.dump
docker cp demand-postgres:/tmp/demand_db.dump "$backupDir\demand_db.dump"
docker exec demand-airflow-db pg_dump -U airflow -d airflow -Fc -f /tmp/airflow.dump
docker cp demand-airflow-db:/tmp/airflow.dump "$backupDir\airflow.dump"
```

Sau backup, chạy `provision-db` với `POSTGRES_ADMIN_URL` đã cấu hình để chỉ tạo
database/role còn thiếu, **chưa tạo bảng**:

```powershell
docker compose --profile tools run --rm pipeline python -m demand_forecasting.pipeline provision-db
```

Sau đó dùng `pg_restore` trên host (hoặc pgAdmin Restore). **Chỉ chạy
`docker compose up -d --build` sau khi restore xong**, để
bootstrap nhìn thấy dữ liệu cũ và không tự seed lịch sử fake mới:

```powershell
pg_restore -h localhost -p 5432 -U demand_user -d demand_db --no-owner --no-acl "$backupDir\demand_db.dump"
pg_restore -h localhost -p 5432 -U airflow -d airflow --no-owner --no-acl "$backupDir\airflow.dump"
```

Nếu tài khoản Airflow cần toàn quyền schema `public`, cấp quyền cho owner
`airflow` trước khi chạy `airflow db migrate`. Kiểm tra số dòng trong
`public.fake_demand_10min` trên host và DAG runs sau khi Airflow khởi động;
chỉ xóa backup/volume cũ khi đã xác nhận dữ liệu được chuyển đầy đủ. Nếu chưa
từng chạy bản cũ thì bỏ qua mục này.

## 10. Dừng và xóa

Giữ nguyên database PostgreSQL trên host, model và output Docker:

```powershell
docker compose down
```

Chỉ xóa volume Docker dành cho **model, Parquet, output, log Airflow** của
Compose mới; **không xóa** database PostgreSQL ngoài Docker:

```powershell
docker compose down -v
```

Lệnh `down -v` vẫn xóa các file model/Parquet/output đã lưu trong các volume
vừa nêu; hãy backup chúng nếu muốn giữ lại. Database cũ trong các volume của
Compose phiên bản trước cũng không được chuyển tự động bởi bản mới.

## Ghi chú production

Stack này là baseline deploy trên một máy bằng Docker Compose. Trước khi public
ra Internet cần thêm reverse proxy HTTPS, secret manager, firewall, backup
PostgreSQL trên host và các volume chứa model, monitoring và giới hạn tài
nguyên container. Airflow UI và PostgreSQL không nên expose công khai trực
tiếp.

Fake generator mỗi 10 phút dùng timestamp + seed cố định nên retry cùng bucket
cho cùng kết quả. Khi có nguồn dữ liệu thật, chỉ cần thay task
`ingest_closed_bucket`; phần feature, model, prediction table và API giữ nguyên.

## Chạy Python trực tiếp trên host — tùy chọn

Docker là luồng chính. Nếu cần debug source mới dùng Python host:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
# Host dùng localhost; biến môi trường này chỉ áp dụng cho terminal hiện tại.
$env:DATABASE_URL = 'postgresql://demand_user:YOUR_PASSWORD@localhost:5432/demand_db'
pytest
```
