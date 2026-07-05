# Project Memory

## آخرین به‌روزرسانی: 2026-07-05

### Product Loader — اصلاحات انجام‌شده

1. **Availability parsing** (`parse_availability`)
   - مقادیر منفی/ناموجود **قبل از** موجود بررسی می‌شوند
   - «ناموجود» دیگر به‌خاطر substring «موجود» available حساب نمی‌شود
   - `0`, `False`, `None`, رشته خالی → `0`
   - مقادیر عددی و متن‌های positive/negative پشتیبانی می‌شوند

2. **`force_url`**
   - `ensure_product_file_exists(..., force_url=True)` فقط از URL دانلود می‌کند
   - در صورت خطا → `RuntimeError` (بدون fallback محلی)
   - `force_url=False` → کشف فایل محلی، سپس fallback دانلود

3. **پشتیبانی CSV و Excel**
   - `.csv` → `pandas.read_csv`
   - `.xlsx` / `.xls` → `pandas.read_excel` (نیاز: `openpyxl`)

4. **کشف فایل محلی**
   - پوشه: `500-پروداکتس` (جستجوی بازگشتی)
   - basename: `products_500`
   - extensions: `.csv`, `.xlsx`, `.xls`
   - fallback: `products_500.csv` در ریشه پروژه

### مسیر داده کشف‌شده

```
<PROJECT_ROOT>/products_500.csv
```

پوشه `500-پروداکتs` در زمان بررسی **موجود نبود**؛ سیستم از فایل ریشه استفاده می‌کند.

### تست‌ها

```bash
python -m pytest tests/test_product_loader.py -v
```

**نتیجه:** 5 passed (2026-07-05)

**یادداشت:** `git diff -- tests/test_product_loader.py` خالی بود — تست‌ها تغییر نکرده‌اند.

### قابلیت‌های قبلی (جلسه قبل)

- `tool_agent_node` با LLM + `bind_tools`
- `send_photo` برای پیشنهاد محصول در تلگرام
- `/reset` واقعی با `MemorySaver.delete_thread`
- `docs/graph.png` + `scripts/generate_graph_png.py`
- `scripts/run_demo_simulation.py` + راهنمای ویدئو در `docs/demo_script.md`

### گام بعدی پیشنهادی

1. فایل محصول را در `500-پروداکتs/products_500.csv` (یا `.xlsx`) قرار دهید
2. ضبط ویدئوی دمو طبق `docs/demo_script.md`
3. اگر تست‌های گسترده‌تر availability/force_url اضافه شد، بدون تغییر تست فقط implementation را بررسی کنید
