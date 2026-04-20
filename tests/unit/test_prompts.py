from __future__ import annotations

from src.rag.prompts import _is_ocr_garbage, build_messages, build_system_prompt, format_context
from src.rag.retriever import Hit
from src.subjects.schema import Subject


def _subject() -> Subject:
    return Subject(slug="theory_of_systems", title_en="Theory of Systems", title_ru="Теория систем")


def _hit(book: str, page: int, text: str) -> Hit:
    return Hit(text=text, subject_slug="theory_of_systems", book=book, page=page, score=0.5)


def test_system_prompt_includes_subject_and_no_think() -> None:
    p = build_system_prompt(_subject())
    # Subject title must be present so the LLM knows what course context it's in.
    assert "Теория систем" in p
    # /no_think disables Qwen3's chain-of-thought — must be first directive.
    assert p.startswith("/no_think")
    # Formatting hints (HTML tags, bullets) so Telegram renders the answer.
    assert "<b>" in p
    assert "•" in p


def test_system_prompt_without_subject_mentions_materials() -> None:
    p = build_system_prompt(None)
    # Without a pinned subject the prompt talks about generic study materials.
    assert "материал" in p.lower()


def test_system_prompt_english_uses_english_title() -> None:
    p = build_system_prompt(_subject(), lang="en")
    assert "Theory of Systems" in p
    assert p.startswith("/no_think")
    # English prompt keeps the same HTML-formatting guidance.
    assert "<b>" in p


def test_context_respects_budget() -> None:
    hits = [_hit("a.pdf", 1, "x" * 2000), _hit("b.pdf", 2, "y" * 2000)]
    ctx = format_context(hits, max_chars=1500)
    assert len(ctx) <= 1500 + 200  # header overhead allowance


def test_context_omits_filename_and_page() -> None:
    """Context must NOT leak the PDF filename or page numbers — both were
    triggering Qwen3-4B to hallucinate fake author attributions like
    «(В. К. Волков, 2005)» from "volkova_v_n_denisov…pdf".
    """
    hits = [_hit("volkova.pdf", 42, "определение системы...")]
    ctx = format_context(hits)
    assert "volkova" not in ctx.lower()
    assert "стр." not in ctx
    assert "page" not in ctx.lower()
    # The chunk text itself must of course survive.
    assert "определение системы" in ctx


def test_build_messages_structure() -> None:
    hits = [_hit("a.pdf", 1, "короткий текст")]
    msgs = build_messages("что такое X?", hits, _subject())
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "что такое X?" in msgs[1]["content"]
    # Filename must NOT be in the user message — see the test above.
    assert "a.pdf" not in msgs[1]["content"]
    # The chunk text body must be there.
    assert "короткий текст" in msgs[1]["content"]


# ---------- OCR garbage filter ----------


def test_ocr_filter_accepts_clean_russian_prose() -> None:
    text = (
        "Стейкхолдер — это физическое или юридическое лицо, которое может "
        "повлиять на достижение организацией своих целей или на работу "
        "организации в целом. Заинтересованные стороны включают акционеров, "
        "поставщиков, клиентов, сотрудников и государственные органы."
    )
    assert not _is_ocr_garbage(text)


def test_ocr_filter_accepts_acronym_compound_words() -> None:
    """STEP-анализ / SWOT-анализ / PESTELанализ are legitimate Russian tokens,
    not mixed-alphabet OCR garbage."""
    text = (
        "Для внешней среды применяют STEP-анализ (PEST-анализ) или его "
        "расширенный вариант — PESTELанализ. Также используется SWOT-анализ "
        "в сочетании с теорией заинтересованных сторон. В. Г. Колосов "
        "предложил альтернативную методику описания среды."
    )
    assert not _is_ocr_garbage(text)


def test_ocr_filter_accepts_text_with_urls() -> None:
    """A legitimate source URL tokenises to many Latin pieces but must not
    flip the 'too much Latin' heuristic."""
    text = (
        "Основные требования к иерархическим структурам: в структурах не "
        "должно быть «вырожденных» ветвей, когда у родительского элемента "
        "отсутствуют дочерние. Источник: "
        "https://elib.spbstu.ru/dl/3/2024/vr/vr24-1408.pdf/info"
    )
    assert not _is_ocr_garbage(text)


def test_ocr_filter_rejects_mixed_alphabet_garbage() -> None:
    """Tesseract mangling of a Russian page: mid-word Latin/Cyrillic mixing."""
    text = (
        "Gow Бергаланфи определил экаифинатьность как pearpasse в мирытых "
        "системах. Wenartpompyeaie noxxon ‘Tor термин предельных состояний "
        "fades ‘oro Mow системы при полностмо начальных условиях зависяще."
    )
    assert _is_ocr_garbage(text)


def test_ocr_filter_rejects_latin_dominant_debris() -> None:
    """When OCR loses a diagram, the chunk ends up mostly Latin gibberish
    with stray Russian. The filter must drop it."""
    text = (
        "„шт == some До ==YASS же И ownsere een conc i emer eter mee Feira "
        "cog es cemmenra eet a me te n nness na эффективность Soe ой риа "
        "ералосеор cere eed ae ea a"
    )
    assert _is_ocr_garbage(text)


def test_format_context_drops_garbage_hits() -> None:
    good = _hit("a.pdf", 1, "Стейкхолдер — это физическое или юридическое лицо, "
                            "которое может повлиять на работу организации.")
    bad = _hit("b.pdf", 2, "Gow Бергаланфи определил pearpasse Wenartpompyeaie "
                            "noxxon Tor термин fades Mow полностмо зависяще.")
    ctx = format_context([good, bad])
    assert "Стейкхолдер" in ctx
    assert "Бергаланфи" not in ctx
    assert "pearpasse" not in ctx


def test_format_context_keeps_one_chunk_when_all_garbage() -> None:
    """Fallback: rather than emit an empty context, keep the single best hit
    — the system prompt's 'suspicious surnames are OCR' rule will catch it."""
    bad1 = _hit("a.pdf", 1, "Gow Бергаланфи pearpasse Wenartpompyeaie noxxon "
                             "Tor термин fades Mow полностмо зависяще conc emer.")
    bad2 = _hit("b.pdf", 2, "„шт == some До ==YASS ownsere een conc i emer "
                             "eter mee Feira cog es cemmenra eet a me te n.")
    ctx = format_context([bad1, bad2])
    assert ctx  # not empty
    # Exactly one chunk, so no separator.
    assert "\n---\n" not in ctx
