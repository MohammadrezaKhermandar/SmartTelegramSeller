# Project Memory

## آخرین به‌روزرسانی: 2026-07-05

### Note-consistency fix + xAI provider (2026-07-05, بعد از restart)

**Symptom:** بعد از restart، pipeline درست بود اما پیام متناقض: «محصولی پیدا نکردم…» و بلافاصله زیرش Lenovo IdeaPad 5 (۳۹,۳۲۵,۰۰۰) نمایش داده می‌شد.

**Root causes:**
1. `hybrid_recommend` وقتی نتایج < `MIN_RECOMMENDATIONS` بود، `SEARCH_NO_MATCH_MESSAGE` را به‌عنوان note برمی‌گرداند در حالی که محصول هم داشت
2. `tool_agent` وقتی note خالی بود، متن دیباگ‌مانند «نتایج بر اساس فیلتر دسته و RAG» را به کاربر نشان می‌داد

**Fix — قانون note:**
- ۰ نتیجه → فقط `SEARCH_NO_MATCH_MESSAGE`، بدون محصول، بدون fallback به دسته دیگر
- ۱ نتیجه → «فقط یک گزینه نزدیک به شرایطت پیدا کردم.» + همان محصول
- ۲ نتیجه → «فقط دو گزینه نزدیک به شرایطت پیدا کردم.» + محصولات
- ۳+ → مقدمه عادی، بدون note
- پیام‌های limited در `SEARCH_LIMITED_MATCH_MESSAGES` (در `prompts.py`) و در `format_product_recommendation` جایگزین مقدمه می‌شوند

**LLM provider:**
- پشتیبانی xAI (Grok) اضافه شد: `LLM_PROVIDER=xai` + `XAI_API_KEY` در `.env` (کلاینت سبک با `requests`، سازگار OpenAI، پروکسی از `LLM_PROXY`/`TELEGRAM_PROXY`)
- خطای دائمی 401/403 (کلید نامعتبر یا نبود credit) → LLM برای کل session خاموش می‌شود؛ فقط یک log warning، بدون retry، بدون چاپ کلید
- ⚠️ کلید xAI فعلی معتبر است ولی تیم credit ندارد → 403 «team doesn't have any credits». تا خرید credit در console.x.ai، ربات با پاسخ آفلاین (بدون polish) کار می‌کند
- کلید Groq قبلی نامعتبر بود و از `.env` حذف شد

**Security:**
- `.env` در `.gitignore` است و track نمی‌شود (بررسی شد: `git ls-files` فقط `.env.example` را نشان می‌دهد)
- هیچ کلیدی در source code یا لاگ‌ها نیست؛ README بخش «امنیت کلیدها» گرفت

#### Changed files

| File | Change |
|------|--------|
| `app/tools/pandas_tools.py` | note بر اساس تعداد نتایج (۰/۱/۲/۳+) |
| `app/graph/prompts.py` | `SEARCH_LIMITED_MATCH_MESSAGES`، مقدمه limited در `format_product_recommendation`، پیشنهاد تغییر دسته در no-match |
| `app/graph/tool_agent.py` | حذف note دیباگ «نتایج بر اساس فیلتر…» |
| `app/llm/client.py` | provider xai + غیرفعال‌سازی دائمی LLM روی 401/403 |
| `app/config.py` | `XAI_API_KEY`, `XAI_MODEL`, `XAI_BASE_URL`, `LLM_PROXY` |
| `.env` | `LLM_PROVIDER=xai` + کلید xAI (فقط لوکال، commit نمی‌شود) |
| `README.md` | جدول env جدید + نکته امنیت کلیدها |
| `tests/test_pandas_tools.py` | تست‌های note برای ۰/۱/۲/۳+ |
| `tests/test_env_security.py` | `.env` ignored/untracked، `.env.example` بدون secret |

#### Tests run

```bash
python -m pytest        # 61 passed
python -m compileall app  # OK
```

#### Remaining issues

1. حساب xAI credit ندارد → polish غیرفعال تا شارژ حساب (پیام‌ها بدون LLM هم کامل‌اند)
2. ربات باید restart شود تا کد جدید load شود
3. فقط یک instance ربات اجرا شود

---

### Recommendation pipeline fix (2026-07-05)

**Symptom:** After `یه لپ‌تاپ می‌خوام` → `بودجه ۱۰۰ میلیون برای برنامه‌نویسی`, bot returned coffee makers, mouse, etc.

**Root causes:**
1. `hybrid_recommend` ranked RAG hits from the **full catalog** and relaxed filters when count was low
2. Search ran **without category** when only budget/usage were in the message (`raw_query` was treated as category substitute)
3. `has_recommendations` in intent detection was `True` whenever `requirements.category` existed → gathering answers routed to `change_preferences`
4. Budget-only messages during gathering triggered `is_preference_update` without actual recommendations

**Fix — mandatory pipeline:**
```
User → Intent → Requirement extraction + merge → Category normalization
→ Pandas hard filters (category, availability, price, brand)
→ RAG scores ONLY candidates that passed filters
→ LLM response / format
```

- No cross-category fallback; empty filtered set → `SEARCH_NO_MATCH_MESSAGE`
- `normalize_category()` maps laptop/notebook → `لپ‌تاپ`
- `usage` (e.g. برنامه‌نویسی) never replaces `category`

#### Changed files

| File | Change |
|------|--------|
| `app/tools/pandas_tools.py` | Rewrite `hybrid_recommend`: filter-first, RAG re-rank only |
| `app/graph/nlp.py` | `normalize_category`, `is_requirement_completion`, intent fixes |
| `app/graph/nodes.py` | Category required for search; fix `has_recommendations` |
| `app/graph/tool_agent.py` | Delegates to strict `hybrid_recommend` |
| `app/graph/prompts.py` | `SEARCH_NO_MATCH_MESSAGE` |
| `app/graph/routers.py` | Category required in post-update routing |
| `tests/test_memory_behavior.py` | Two-turn laptop gathering test |
| `tests/test_pandas_tools.py` | No category / cross-category tests |

#### Tests run

```bash
python -m pytest
python -m compileall app
```

**Result:** 54 passed (2026-07-05)

#### Remaining issues

1. Groq 403/429 when `USE_LLM=true` — hybrid pipeline still works via fallback
2. Only one Telegram bot instance should run
3. Git push manual

#### Next step

Restart bot and verify two-turn laptop demo in Telegram

---

### Previous fixes (2026-07-05)

- LangGraph msgpack serialization (`to_json_safe`)
- Budget update preserves category (`merge_requirements`, `apply_filters` chain)
- Demo docs + README Demo Notes

---

### Product Loader / earlier features

- Availability parsing, CSV/Excel, `force_url`, local discovery
- Tool agent, send_photo, `/reset`, graph.png, demo scripts
