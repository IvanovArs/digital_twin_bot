"""Валидация преподавательских аплоадов.

Две независимые проверки:

* **Лимит размера** — отбивает файлы выше разумного порога *до* download'а
  и записи на диск. Telegram-bot технически принимает до 2 ГБ через
  ``bot.download``; без лимита один upload забил бы сервер.

* **Magic-byte sniffing** — расширение файла контролируется клиентом,
  вредоносный/мис-лейбленный файл могут переименовать в ``.pdf`` и
  проскочить allowlist по расширению. Проверяем, что первые байты
  соответствуют заявленному типу. Plain-text форматы (``.txt``, ``.md``,
  ``.csv``, ``.yaml``) проверяем на utf-8-декодируемость.
"""

from __future__ import annotations

# Щедрый лимит: покрывает самые большие учебники (~30 МБ scan'ы с
# картинками), но не даёт лить disk-image'ы.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def check_size(size: int | None) -> str | None:
    """None — если размер ок, иначе человекочитаемая ошибка."""
    if size is None:
        return None  # размер неизвестен — caller решает
    if size <= 0:
        return "Пустой файл."
    if size > MAX_UPLOAD_BYTES:
        return (
            f"Файл слишком большой: {size / 1_048_576:.1f} МБ. "
            f"Лимит — {MAX_UPLOAD_BYTES // 1_048_576} МБ."
        )
    return None


def _is_pdf(head: bytes) -> bool:
    return head.startswith(b"%PDF-")


def _is_docx(head: bytes) -> bool:
    # DOCX — это ZIP-архив. PK\x03\x04 — стандартный local-file header;
    # PK\x05\x06 / PK\x07\x08 встречаются на edge-cases (пустой/spanned).
    return head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _is_utf8_text(payload: bytes) -> bool:
    # Короткой проверки достаточно — если первые 4 КБ декодируются,
    # остальное тоже декодируется (или прилетит bad-byte ниже — будет warning).
    try:
        payload[:4096].decode("utf-8-sig")
        return True
    except UnicodeDecodeError:
        return False


def check_magic(filename: str, payload: bytes) -> str | None:
    """Проверить, что содержимое файла соответствует расширению.

    None — если ок, иначе человекочитаемая ошибка. Текст ошибки упоминает
    заявленное расширение, не реальный content-type — не сливаем наружу,
    что мы знаем, что .pdf на самом деле ZIP (в других контекстах это
    dual-use information disclosure).
    """
    if not payload:
        return "Пустой файл."
    name = filename.lower()
    head = payload[:16]
    if name.endswith(".pdf"):
        if not _is_pdf(head):
            return "Файл с расширением .pdf не является PDF."
        return None
    if name.endswith(".docx"):
        if not _is_docx(head):
            return "Файл с расширением .docx не является DOCX."
        return None
    if name.endswith((".txt", ".md", ".csv", ".yaml", ".yml")):
        if not _is_utf8_text(payload):
            return "Текстовый файл не в UTF-8."
        return None
    # Неизвестные расширения должен отбить allowlist в handler'е выше;
    # если попали сюда — отказываем по принципу fail-closed.
    return "Неподдерживаемое расширение."
