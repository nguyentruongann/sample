# Đối chiếu notebook và kiểm tra luồng vận hành

## Hai phép đo khác nhau

`notebooks/source_main.ipynb` chia 80% timeline gốc thành mốc mở đầu test,
30 ngày trước đó là validation. Dữ liệu gốc chạy từ 01/05 đến 11/09/2026;
test H10 bắt đầu 15/08/2026 09:40 (giờ Việt Nam) và kết thúc trước 11/09.
Lệnh `pipeline train` (không có `--rolling`) tái hiện cửa sổ này, ghi
`outputs/notebook_benchmark_summary.csv`, chỉ đọc phần lịch sử gốc và không
thay model serving.

Bootstrap và DAG retrain dùng train 90 ngày, validation 30 ngày và test 30
ngày gần nhất. Từ 11/09 trở đi, test rolling bao gồm dữ liệu được generator
live sinh tiếp. Vì dữ liệu và cửa sổ train khác nhau, không so trực tiếp WMAPE
rolling với các số 15,63% và 14,66% của test H10 trong notebook.

## Kiểm tra trên dữ liệu tái tạo cục bộ

Tạo lại 5.404.530 dòng lịch sử bằng `fake_data.py` trích từ notebook, sinh
bucket live 10 phút từ 11/09 đến 26/09/2026 00:00 UTC bằng
`incremental_data.py`, sau đó dựng feature và train LightGBM CPU với cấu hình
mặc định (300.000 train/100.000 validation để tìm số vòng và số lá, rồi refit
trên toàn bộ train + validation). Bảng dưới là kiểm tra **dữ liệu giả cục bộ**,
không phải kết quả đo trên PostgreSQL của người dùng. Trên máy khác hoặc dữ
liệu đã thay đổi, WMAPE có thể khác.

| Mode | Horizon | Validation WMAPE | Test rolling WMAPE | Test live WMAPE |
|---|---:|---:|---:|---:|
| Car | H10 | 15,62% | 15,71% | 15,57% |
| Car | H30 | 16,44% | 16,42% | 16,20% |
| Car | H60 | 20,07% | 20,01% | 19,79% |
| Motorcycle | H10 | 15,18% | 14,87% | 15,16% |
| Motorcycle | H30 | 15,94% | 15,53% | 15,75% |
| Motorcycle | H60 | 19,58% | 19,05% | 19,30% |

Cả sáu model đạt WMAPE test dưới ngưỡng promotion 30% trong phép kiểm tra này.
`models/active.json` được tạo khi và chỉ khi đủ bộ sáu model hợp lệ.

Khi nâng cấp từ generator `notebook-calibrated-v3`, bootstrap tạo lại **chỉ
những dòng giả live từ 11/09** và retrain vì schema model mới. Lịch sử gốc
trước 11/09 được giữ; lần khởi động sau sẽ kiểm tra phiên bản để tránh train
lại. Test unit/integration Python: 34 passed.
