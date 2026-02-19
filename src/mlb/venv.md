Отличная идея 💪
Сделаем **короткую, практичную шпаргалку по venv для Windows** — отдельно для **PowerShell** и **Git Bash**, с блоком “если что-то сломалось”.

Сохрани это в `docs/venv_cheatsheet.md` — пригодится 100%.

---

# 🐍 VENV CHEATSHEET (Windows)

Проект: `ml-blueprint`
Python: 3.10+

---

# 🟦 PowerShell (Рекомендуется)

## 🔹 1. Создать окружение

```powershell
cd C:\Users\Shuhrat\VsCode\ml-blueprint
python -m venv .venv
```

---

## 🔹 2. Активировать

```powershell
.\.venv\Scripts\Activate.ps1
```

Если ошибка про Execution Policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Закрыть PowerShell → открыть снова → активировать.

---

## 🔹 3. Проверить, что всё ок

```powershell
python -c "import sys; print(sys.executable)"
```

Ожидаемый путь:

```
...\ml-blueprint\.venv\Scripts\python.exe
```

---

## 🔹 4. Установить проект

```powershell
pip install -e ".[dev]"
```

---

## 🔹 5. Запуск тестов (без путаницы)

Всегда так:

```powershell
python -m pytest
```

Не просто `pytest`.

---

## 🔹 6. Деактивировать

```powershell
deactivate
```

---

# 🟩 Git Bash

⚠️ Частая причина проблем — Git Bash использует другой Python.

---

## 🔹 1. Создать окружение

```bash
python -m venv .venv
```

---

## 🔹 2. Активировать

```bash
source .venv/Scripts/activate
```

---

## 🔹 3. Проверить Python

```bash
which python
python -c "import sys; print(sys.executable)"
```

Должен указывать на `.venv`.

---

## 🔹 4. Установить проект

```bash
python -m pip install -e ".[dev]"
```

---

## 🔹 5. Запуск тестов

```bash
python -m pytest
```

---

# 🛑 Если что-то сломалось

## ❌ Ошибка: ModuleNotFoundError: mlb.core

Причина: pytest запущен не из venv.

Решение:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

---

## ❌ pytest использует системный Python

Проверь:

```powershell
where python
where pytest
```

Если путь не из `.venv` → не активировано окружение.

---

## ❌ Ruff проверяет весь компьютер

Ты не в папке проекта.

Проверь:

```powershell
pwd
```

Или запускай:

```powershell
ruff check src tests
```

---

## ❌ Всё совсем странно

Самый быстрый reset:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

---

# 🧠 Лучший рабочий алгоритм

Каждый раз, когда открываешь терминал:

```powershell
cd C:\Users\Shuhrat\VsCode\ml-blueprint
.\.venv\Scripts\Activate.ps1
```

И только потом работаешь.

---

# 🏆 Профессиональный лайфхак

Чтобы не забывать активировать venv:

В VS Code:

1. Ctrl+Shift+P
2. "Python: Select Interpreter"
3. Выбрать `.venv`

Тогда терминал будет автоматически использовать правильный Python.

---

Если хочешь, могу сделать:

* 📦 версию этой шпаргалки как готовый `.md` файл
* или сделать `make reset-env` скрипт для быстрого восстановления

Готов перейти к Day 3 (Config + Logging + Pydantic)?
