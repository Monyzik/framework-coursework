# Курсовой проект

Исследуем влияние ручного и искусственного обогащения данных.

## Данные

- **Fashion** — 4000 фотографий: футболки и топы, брюки,
  платья. [Скачать набор](https://huggingface.co/datasets/Monyzik/fashion-4000).
- **RuSpellGold** — 1711 пар «текст —
  исправление». [Источник](https://huggingface.co/datasets/RussianNLP/RuSpellGold/blob/8b6694e45ac9222bc705ee4af9d6367ac083ff5c/data/complete_test/test.json).

Данные сохраняются локально:

- `data/raw/imaterialist/` — описания классов и папка `train/` с фото.
- `data/raw/ruspellgold/test.jsonl` — текстовый датасет, одна JSON-запись на строку.
- `data/processed/` — список выбранных фото и их аннотации.

RuSpellGold опубликован только как тестовый набор. Разбиения для обучения ещё нужно определить.

Fashion работает с готовой выборкой из 4000 фото. Если есть полный `train.csv`, ноутбук использует его для анализа исходного набора.

## Запуск

Нужен Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Откройте `notebooks/fashion.ipynb` или `notebooks/text.ipynb` и выполните ячейки по порядку. Ноутбуки установят недостающие библиотеки и скачают данные. Готовые локальные файлы используются повторно.

Скрипты: `scripts/install_dependencies.py` и `scripts/download_data.py`.

## Разметка текста

Каждый разметчик запускает Docker на своём компьютере и выполняет `docker compose up -d`
из папки проекта. Откройте `http://localhost:8080` и создайте один проект для текста.
Проекты и разметка хранятся в томе Docker. Остановка через `docker compose down` сохраняет данные.

Вставьте этот XML в настройки проекта **Labeling Interface → Code**.

```xml
<View>
  <Style>.htx-text { white-space: pre-wrap; }</Style>
  <Text name="source" value="$text" />
  <TextArea name="correction" toName="source" rows="5" editable="true"
            required="true" maxSubmissions="1" placeholder="Вставьте полный текст" />
</View>
```
