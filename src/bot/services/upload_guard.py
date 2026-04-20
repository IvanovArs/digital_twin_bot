"""Validation helpers for teacher-uploaded files.

Two independent checks:

* **Size cap** — rejects files above a sensible limit *before* we pay
  the cost of downloading + hitting disk. Telegram bots can technically
  receive up to 2 GB payloads via ``bot.download``; without a cap here
  an admin could fill the server with a single upload.

* **Magic-byte sniffing** — the file extension is client-controlled, so a
  malicious or mis-labeled file can be renamed to ``.pdf`` and slip
  through the extension whitelist. We verify the first few bytes match
  what the extension claims. Plaintext formats (``.txt``, ``.md``,
  ``.csv``, ``.yaml``) get a utf-8-decodability check instead.
"""

from __future__ import annotations

# Generous cap: covers the biggest textbook PDFs we've seen (~30 MB
# scanned with images) but stops anyone dumping a disk image.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def check_size(size: int | None) -> str | None:
    """Return None if the size is acceptable, else a user-facing error."""
    if size is None:
        return None  # unknown size — let the caller decide
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
    # DOCX is a ZIP archive. PK\x03\x04 is the standard local-file header;
    # PK\x05\x06 / PK\x07\x08 appear on edge-cases (empty archive, spanned).
    return head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _is_utf8_text(payload: bytes) -> bool:
    # Short probe is enough — if the first 4 KB decode, the rest will too
    # (or we'll hit a bad byte downstream and emit a soft warning).
    try:
        payload[:4096].decode("utf-8-sig")
        return True
    except UnicodeDecodeError:
        return False


def check_magic(filename: str, payload: bytes) -> str | None:
    """Verify the file content matches its extension.

    Returns None if the file passes, a user-facing error string otherwise.
    The error mentions the claimed extension, not the content type — we
    don't want to leak back that we know a PDF is actually a ZIP (that's a
    dual-use information disclosure in other contexts).
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
    # Unknown extensions should have been rejected upstream by the
    # handler's extension allowlist; if we land here, fail closed.
    return "Неподдерживаемое расширение."
