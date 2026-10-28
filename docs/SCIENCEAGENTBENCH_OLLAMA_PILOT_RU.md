# ScienceAgentBench + Ollama — статус пилота

## Цель

Проверить путь генерации программ локальной моделью на официальном split `verified`, не меняя upstream benchmark code и не исполняя сгенерированные программы на этапе генерации.

## Конфигурация

- Dataset: `osunlp/ScienceAgentBench`
- Split: `verified` (102 задачи)
- Local model: `qwen2.5-coder:7b`
- Inference endpoint: Ollama, `http://localhost:11434`
- Adapter: v4, прямые запросы к Ollama с минимальным in-memory compatibility shim вместо недоступной pinned версии LiteLLM
- Runtime: Python 3.10 virtual environment
- Generation-only mode: ON

## Пилот 2026-10-28

| Задача | Время генерации | Файл создан |
|---|---:|---|
| `clintox_nn.py` | 10,7 с | Да |
| `mat_feature_select.py` | 6,0 с | Да |
| `predict_bulk_modulus.py` | 7,1 с | Да |
| **Итого** | **23,8 с** | **3/3** |

Статическая проверка обнаружила 3 непустые программы, успешно прошедшие Python AST parse. Проверены JSONL trace (3 записи), manifest и совпадение task names с первыми тремя строками verified split. Для staging были созданы копии с именами `pred_clintox_nn.py`, `pred_mat_feature_select.py`, `pred_predict_bulk_modulus.py`; SHA-256 исходника и копии совпал.

## Что это НЕ доказывает

- Сгенерированные программы не исполнялись.
- Нет проверенных результатов выполнения и нет official score.
- AST parse не проверяет зависимости, корректность ML pipeline, качество предсказаний или соответствие научной задаче.
- Три задачи — пилот интеграции, а не репрезентативная оценка модели.

## Блокер официальной оценки

На момент отчёта в локальном clone не обнаружены официальный `benchmark_verified.zip` и требуемые benchmark subdirectories (`datasets`, `eval_programs`, `gold_programs`, `scoring_rubrics`). Следующий шаг — получить артефакты из официального источника ScienceAgentBench, проверить архив и Docker evaluation environment. Не использовать сторонние копии и не публиковать архив/эталонные данные.

## Официальные ссылки

- [README / evaluation instructions](https://github.com/OSU-NLP-Group/ScienceAgentBench)
- [Verified dataset on Hugging Face](https://huggingface.co/datasets/osunlp/ScienceAgentBench)
