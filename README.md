# RUAccent

RUAccent — библиотека для автоматической расстановки ударений (и восстановления «ё») в русском тексте.
Форк [Den4ikAI/ruaccent](https://github.com/Den4ikAI/ruaccent), доведённый до состояния, пригодного для
продакшена на CPU: без записи в `site-packages`, с воспроизводимой версией моделей, контролем потоков
ONNX Runtime, кешем предложений и недеструктивным выводом (в тексте появляются только `+` и «ё»).

**По вопросам коммерческого использования моделей пишите автору оригинальной библиотеки: [telegram](https://t.me/bceloss).**

## Установка

```
pip install git+https://github.com/NikiPshg/ruaccent@<commit>
```

Зависимости: `onnxruntime`, `transformers`, `tokenizers`, `numpy`, `huggingface_hub`, `razdel`.
Для CUDA поставьте `onnxruntime-gpu` (`pip install "ruaccent[gpu] @ ..."`) и передайте `providers`.

## Использование

```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(
    omograph_model_size="turbo3.1",   # tiny | tiny2 | tiny2.1 | turbo | turbo2 | turbo3 | turbo3.1 | small_poetry | medium_poetry | big_poetry
    use_dictionary=True,              # полный словарь ударений (~+0.7 ГБ RAM), иначе только нейросети
    tiny_mode=False,                  # True: без предсказателя «нужно ли ударение» и без словаря
    num_threads=1,                    # потоки ONNX Runtime на сессию (см. ниже)
    workdir="/models/ruaccent",       # куда скачать модели; по умолчанию ~/.cache/ruaccent
)

print(accentizer.process_all("на двери висит замок."))
# на двер+и вис+ит зам+ок.
```

### Параметры `load`

| параметр | по умолчанию | смысл |
|---|---|---|
| `omograph_model_size` | `turbo2` | модель снятия омографии (`turbo3.1` — лучшая, 368 МБ; `tiny2.1` — 43 МБ) |
| `use_dictionary` | `False` | полный словарь ударений вместо `accents_nn` |
| `custom_dict` | `None` | свои ударения: `{"слово": "сл+ово"}` |
| `custom_homographs` | `None` | свои омографы: `{"замок": ["з+амок", "зам+ок"]}` |
| `providers` | `["CPUExecutionProvider"]` | execution providers ONNX Runtime, например `[("CUDAExecutionProvider", {"device_id": 0})]` |
| `num_threads` | `None` (все ядра) | `intra_op_num_threads` каждой сессии. **В проде ставьте `1`**: задержка на предложение та же (~7 мс на M-серии), CPU меньше в 10+ раз |
| `session_options` | `None` | готовый `onnxruntime.SessionOptions` вместо `num_threads` |
| `repo` / `revision` | `ruaccent/accentuator` @ закреплённый коммит | откуда и какую ревизию моделей качать; `revision="main"` — всегда последние |
| `workdir` | `$RUACCENT_WORKDIR` → `~/.cache/ruaccent` | каталог моделей; пакет никогда не пишет в свою папку |
| `tiny_mode` | `False` | облегчённый режим |
| `token` | `None` | токен Hugging Face (репозиторий моделей публичный) |
| `local_files_only` | `False` | не ходить в сеть; упасть, если моделей нет |

Модели скачиваются один раз через `huggingface_hub.snapshot_download` (докачка при обрыве, проверка по
метаданным). Если при следующем запуске сети нет, а файлы на месте — библиотека работает с локальной копией.

### Кеш

Результат для каждого предложения кешируется (`functools.lru_cache`, по умолчанию 4096 записей,
`RUAccent(cache_size=...)`). Ключ — предложение без окружающих пробелов, поэтому одна и та же фраза в
начале текста и внутри него даёт попадание. Кеш потокобезопасен; `process_all` можно звать из нескольких
потоков параллельно (ONNX Runtime сессии потокобезопасны). `accentizer.cache_info()` / `accentizer.clear_cache()`.

### Что делает с текстом

`process_all` возвращает исходный текст, в который добавлены `+` перед ударной гласной и восстановлена «ё».
Пунктуация, цифры, латиница, символы (`% № " … – /`), пробелы и переносы сохраняются как есть; удаляются
только управляющие и zero-width символы. Слова, уже содержащие `+`, не трогаются — можно подавать текст
с ручными ударениями. `skip_regex` защищает совпавшие фрагменты от обработки целиком.

`process_yo(text)` — только восстановление «ё».

## Ресурсы (CPU, turbo3.1)

| конфигурация | RSS | загрузка |
|---|---|---|
| `use_dictionary=True` | ~2.0 ГБ | ~3 с |
| `use_dictionary=False` | ~1.3 ГБ | ~2 с |
| `tiny_mode=True` | ~0.8 ГБ | <1 с |

Холодное предложение — 5–7 мс на одном потоке, попадание в кеш — 0.05 мс.

## Отличия от upstream (1.6.0)

- модели и словари качаются в `workdir`, а не в `site-packages`; работает от непривилегированного пользователя;
- закреплённая ревизия моделей, `snapshot_download` вместо пофайлового скачивания;
- убраны неиспользуемые rule-engine и koziev (−200 МБ на диске, −550 МБ RAM, −`python-crfsuite`);
- `num_threads` / `session_options` для ONNX Runtime;
- недеструктивный вывод: больше не вырезаются `" % № … – + /` и т. п., не схлопываются пробелы перед скобками;
- исправлено смещение предсказаний ударений/«ё» после слитной пунктуации (`», ),`);
- кеш на инстанс, ключ без окружающих пробелов, `cache_info()`/`clear_cache()`;
- `pyproject.toml` как единственный источник метаданных, тесты (`pytest`, `pytest --models`), CI.

Файлы моделей и словарей: [huggingface.co/ruaccent/accentuator](https://huggingface.co/ruaccent/accentuator).
