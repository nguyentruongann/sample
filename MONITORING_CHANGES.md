# Các file thêm/sửa — monitoring theo mentor

Gói chỉ gồm file thay đổi, áp dụng lên source demand_forecasting(1).zip. File ngoài danh sách giữ nguyên. Không chứa .env, model, dữ liệu hoặc thông tin đăng nhập.

## Kiểm thử

- 54 test đạt; 1 test tích hợp PostgreSQL skip do chưa có MONITOR_TEST_DATABASE_URL.
- Có test LightGBM thật cho 6 model và khớp feature notebook; reference sinh lúc train và tạo lại từ snapshot cũ có bins/counts giống nhau.
- Có test khung 05:00 → [03:00,04:00), interval 2 giờ, qua ngày, H60 maturity, missing actual, retry, tách version, PSI constant/ngoài range.
- Replay được kiểm tra điều phối bằng mock I/O: schema test riêng, 18 mốc dự báo trong 3 giờ, không có dữ liệu tương lai ở từng mốc, model gốc không đổi.
- Build wheel thành công; kiểm tra HTML tồn tại trong wheel và GET / từ package đã đóng gói trả HTTP 200.
- Python/JavaScript hợp lệ; YAML/JSON Compose, Prometheus và Grafana parse thành công.
- Chưa chạy end-to-end PostgreSQL/Airflow/Docker hoặc replay với toàn bộ model và dữ liệu trên máy bạn. Đã kèm test PostgreSQL tùy chọn.

## Thứ tự nâng cấp

1. Chép file theo đường dẫn trong gói.
2. Build/cập nhật service theo MONITORING.md.
3. Với model cũ: chạy prediction_reference bằng snapshot feature gốc.
4. Bật DAG demand_model_monitoring.
5. Tùy chọn: chạy monitoring_demo --hours 3 và mở URL demo được in.

## Danh sách

| Đường dẫn | Thao tác |
|---|---|
| `Dockerfile` | Thay thế |
| `MONITORING.md` | Thêm mới |
| `README.md` | Thay thế |
| `dags/monitoring_dag.py` | Thêm mới |
| `docker-compose.monitoring.yml` | Thêm mới |
| `monitoring/alerts.yml` | Thêm mới |
| `monitoring/grafana/dashboards/demand.json` | Thêm mới |
| `monitoring/grafana/provisioning/dashboards/demand.yml` | Thêm mới |
| `monitoring/grafana/provisioning/datasources/prometheus.yml` | Thêm mới |
| `monitoring/prometheus.yml` | Thêm mới |
| `requirements-api.txt` | Thay thế |
| `src/demand_forecasting/dashboard_assets/monitoring.html` | Thêm mới |
| `src/demand_forecasting/database.py` | Thay thế |
| `src/demand_forecasting/feature_monitor.py` | Thêm mới |
| `src/demand_forecasting/monitoring.py` | Thêm mới |
| `src/demand_forecasting/monitoring_config.py` | Thêm mới |
| `src/demand_forecasting/monitoring_demo.py` | Thêm mới |
| `src/demand_forecasting/monitoring_service.py` | Thêm mới |
| `src/demand_forecasting/monitoring_store.py` | Thêm mới |
| `src/demand_forecasting/performance_monitor.py` | Thêm mới |
| `src/demand_forecasting/predict.py` | Thay thế |
| `src/demand_forecasting/prediction_reference.py` | Thêm mới |
| `src/demand_forecasting/train.py` | Thay thế |
| `tests/test_final_notebook_contract.py` | Thay thế |
| `tests/test_monitoring.py` | Thêm mới |
| `tests/test_monitoring_postgres.py` | Thêm mới |
| `MONITORING_CHANGES.md` | Thêm mới |
