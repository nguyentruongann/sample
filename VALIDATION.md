# Kiểm chứng bản chuyển từ final.ipynb

## Kết quả đã thực sự chạy

`python -m pytest --disable-warnings --maxfail=3`:
**39 passed, 25 warnings, 18.36 giây** trên môi trường kiểm tra local.
Sau đó chỉ chỉnh cờ báo cáo `hit_tree_limit` để áp dụng riêng cho search và tài liệu;
không thay logic fit/predict đã được kiểm tra.

- Chạy trực tiếp code gốc cell 8/9/11 của notebook từ fixture, rồi so feature,
  target và split với pipeline ở cả 6 tổ hợp mode × horizon.
- Dữ liệu parity có hex khác nhau giữa hai mode, bucket thiếu, hàng xóm thiếu,
  ngày lễ và target tổng 30/60 phút. So cả thứ tự 25/33 feature, NaN và hex-code.
- So feature online với offline trên cùng forecast time. Thay nhãn tại T/tương
  lai không làm thay đổi feature online.
- Fit thật LightGBM cho 6 model `locked`, với số cây/lá đúng bảng notebook;
  kiểm tra refit 6 lần, lưu artifact, feature count, rounding và metric TEST.
- Test `notebook_search` với budget nhỏ để kiểm tra train→save→serve, gọi
  LightGBM thật, không chỉ mock estimator. Không lấy mẫu train/VAL dù legacy
  settings SEARCH_*_ROWS nhỏ hơn dữ liệu.
- Kiểm tra historical train không tự cắt còn 90 ngày.
- Các test có sẵn về target/lag, split/purge, promotion, bootstrap, API/dashboard,
  live generator, DB orchestration và stale inference cũng đã chạy.

Môi trường test: Python 3.12, LightGBM 4.7.0, pandas 2.2.3, NumPy 2.3.5,
scikit-learn 1.8.0. Một số phiên bản khác requirements Docker đã pin; bộ test
không chứng minh tái lập bit-for-bit giữa các môi trường. Không đổi danh sách
requirements của project trong bản sửa này.

## Chưa chạy / không được diễn giải thành kết quả đã chạy

- Chưa train lại toàn bộ hơn 5 triệu dòng của bạn.
- Chưa kết nối PostgreSQL của bạn; các test DB/API dùng dữ liệu nhỏ hoặc mock.
- Chưa build/start Docker Compose hoặc Airflow end-to-end.
- Chưa đánh giá chất lượng trên dữ liệu thực tế hoặc đo latency/RAM production.

Do đó không gán WMAPE notebook cho run pipeline mới. Các số lá/cây cố định lấy
trực tiếp từ output notebook, còn metric mới chỉ có sau khi bạn chạy flow.
Các số WMAPE rolling của bản source cũ không còn được coi là chứng cứ cho
feature schema 25/33 hiện tại.

## Tiêu chí khi chạy trên máy của bạn

1. Log có 6 model và feature_count lần lượt 25/33/33 cho mỗi mode.
2. `locked`: bảng số lá/cây khớp NOTEBOOK_MIGRATION, parameter_source là
   `final_notebook_locked`; `notebook_search`: có 18 trial và 6 refit.
3. So `WMAPE_%` integer với notebook chỉ khi cùng dữ liệu, cutoff, seed và
   environment; rolling dùng cửa sổ khác nên không ép metric phải bằng nhau.
4. Model run có manifest, hex_code_mapping, training_origin và feature_schema;
   inference không nhận artifact schema cũ.
5. Model vẫn là ứng viên demo/shadow: kết luận notebook về lỗi tại các đợt
   tăng/giảm đột ngột không được giải quyết chỉ bằng chuyển source train.
