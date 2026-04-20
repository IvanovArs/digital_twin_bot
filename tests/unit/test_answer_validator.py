"""Tests for the post-LLM attribution validator."""

from __future__ import annotations

from src.rag.answer_validator import validate_answer


def test_grounded_attribution_is_kept() -> None:
    """If the surname AND year are in the retrieved corpus, keep the bracket."""
    answer = "<b>Онтология</b> — формальная спецификация (T. Gruber, 1993)."
    corpus = [
        "Gruber (1993) ввёл термин «онтология» как формальную спецификацию "
        "концептуализации предметной области."
    ]
    out, report = validate_answer(answer, corpus)
    assert "(T. Gruber, 1993)" in out
    assert report.total == 0


def test_hallucinated_attribution_is_stripped() -> None:
    """Bernaldi doesn't exist in the corpus — strip it."""
    answer = "<b>Эмерджентность</b> — системное свойство (G. Bernaldi, 1984)."
    corpus = [
        "Gow Бергаланфи определил эмерджентность как свойство, не сводимое "
        "к свойствам частей."
    ]
    out, report = validate_answer(answer, corpus)
    assert "Bernaldi" not in out
    assert "1984" not in out
    assert "(G. Bernaldi, 1984)" in report.attributions_stripped[0]


def test_grounded_russian_initials_kept() -> None:
    """The prompt's own example — «(И. Фамилия, 1984)» — must survive when
    the fragment actually contains that attribution."""
    answer = "Теория заинтересованных сторон (Р. Е. Фриман, 1984)."
    corpus = ["По Р. Е. Фриману (1984), стейкхолдер — это лицо или группа…"]
    out, report = validate_answer(answer, corpus)
    assert "Фриман" in out
    assert "1984" in out
    assert report.total == 0


def test_hallucinated_initials_stripped() -> None:
    answer = "Системный анализ (В. К. Волков, 2005)."
    corpus = ["Основы теории систем применяются в экономике и менеджменте."]
    out, report = validate_answer(answer, corpus)
    assert "Волков" not in out
    assert "2005" not in out
    assert report.attributions_stripped


def test_grounded_etymology_kept() -> None:
    answer = "<b>Эмерджентность</b> (лат. emergere — выныривать) — свойство."
    corpus = ["От латинского emergere (выныривать), то есть появляться."]
    out, report = validate_answer(answer, corpus)
    assert "(лат. emergere — выныривать)" in out
    assert report.total == 0


def test_hallucinated_etymology_stripped() -> None:
    """«(лат. emergent — взрастающий)» — emergere isn't «взрастать», and the
    corpus doesn't contain «emergent», so the bracket is dropped."""
    answer = "Эмерджентность (лат. emergent — взрастающий) — это свойство."
    corpus = ["Эмерджентность — свойство системы, не сводящееся к свойствам частей."]
    out, report = validate_answer(answer, corpus)
    assert "emergent" not in out
    assert "взрастающий" not in out
    assert report.etymologies_stripped


def test_hallucinated_cjk_stripped() -> None:
    answer = "Энтропия 熵 — мера неопределённости."
    corpus = ["Энтропия — мера неопределённости системы."]
    out, report = validate_answer(answer, corpus)
    assert "熵" not in out
    assert report.foreign_scripts_stripped


def test_grounded_cjk_kept() -> None:
    """If the course material legitimately contains CJK (rare but possible),
    keep it verbatim."""
    answer = "Термин 熵 введён в китайской литературе."
    corpus = ["Энтропия (китайское 熵) — мера неопределённости."]
    out, report = validate_answer(answer, corpus)
    assert "熵" in out
    assert not report.foreign_scripts_stripped


def test_answer_without_attributions_is_untouched() -> None:
    answer = "<b>Система</b> — совокупность взаимосвязанных элементов."
    corpus = ["Система это совокупность элементов."]
    out, report = validate_answer(answer, corpus)
    assert out == answer
    assert report.total == 0


def test_residue_cleanup_after_strip() -> None:
    """After removing a bracket we shouldn't leave a double space or orphan
    punctuation behind."""
    answer = "Эмерджентность (G. Bernaldi, 1984) — это свойство."
    corpus = ["Эмерджентность — свойство системы."]
    out, _ = validate_answer(answer, corpus)
    # No double space, no stranded "  —":
    assert "  " not in out
    assert out == "Эмерджентность — это свойство."


def test_report_object_totals() -> None:
    answer = (
        "Термин (G. Bernaldi, 1984), этимология (лат. fake — нечто) и 文字 здесь."
    )
    corpus = ["Термин описывает некоторое явление."]
    _, report = validate_answer(answer, corpus)
    # 1 attribution + 1 etymology + 1 CJK + 1 year (1984 inside attribution
    # also gets flagged separately by the bare-year pass — it was never
    # in the corpus). Total must be at least the three bracket types.
    assert report.total >= 3
    assert report.attributions_stripped
    assert report.etymologies_stripped
    assert report.foreign_scripts_stripped


# ---------- bare-year validation ----------


def test_grounded_bare_year_survives() -> None:
    answer = "Теория систем была сформулирована в 1968 году."
    corpus = ["Работы Берталанфи 1968 года заложили фундамент общей теории систем."]
    out, report = validate_answer(answer, corpus)
    assert "1968" in out
    assert report.years_stripped == []


def test_ungrounded_bare_year_is_stripped() -> None:
    answer = "Эмерджентность была введена в 1984 году как термин."
    corpus = ["Эмерджентность — свойство системы, не сводимое к сумме свойств частей."]
    out, report = validate_answer(answer, corpus)
    assert "1984" not in out
    assert "1984" in report.years_stripped


def test_year_strip_cleans_russian_scaffolding() -> None:
    """After stripping «1984», the bare «в  году» leftover is ugly —
    the cleanup rule must take the whole scaffolding."""
    answer = "Термин появился в 1999 году и стал общепринятым."
    corpus = ["Термин связан с развитием теории систем."]
    out, _ = validate_answer(answer, corpus)
    assert "году" not in out
    # The surrounding sentence is still grammatical:
    assert out.startswith("Термин появился")


def test_year_strip_leaves_other_numbers_intact() -> None:
    """Only 4-digit years from 1500-2029 get stripped; any other number
    (chapter ref, page, percentage) must survive."""
    answer = "См. главу 7, стр. 42 — 80% случаев покрыты."
    corpus = ["Глава 7 описывает 80% случаев на странице 42."]
    out, _ = validate_answer(answer, corpus)
    assert "7" in out and "42" in out and "80" in out


def test_year_validation_survives_empty_corpus() -> None:
    """No chunks → everything is ungrounded → years go away, no crash."""
    answer = "Документ принят в 2015 году."
    out, report = validate_answer(answer, [])
    assert "2015" not in out
    assert "2015" in report.years_stripped


# ---------- decade / date-range validation ----------


def test_ungrounded_decade_is_stripped() -> None:
    answer = "Методика разрабатывалась в 1970-х годах."
    corpus = ["Методика — это способ организации исследования."]
    out, report = validate_answer(answer, corpus)
    assert "1970-х" not in out
    assert report.decades_stripped


def test_grounded_decade_survives() -> None:
    answer = "Термин появился в 1970-х."
    corpus = ["В 1970-х годах была разработана общая теория систем."]
    out, report = validate_answer(answer, corpus)
    assert "1970-х" in out
    assert not report.decades_stripped


def test_ungrounded_date_range_is_stripped() -> None:
    answer = "Активные исследования шли в 1970-1990-е годы."
    corpus = ["Исследования в области теории систем."]
    out, report = validate_answer(answer, corpus)
    assert "1970-1990" not in out
    assert report.decades_stripped


def test_grounded_date_range_survives_when_both_years_present() -> None:
    answer = "Развитие пришлось на 1970-1990 годы."
    corpus = [
        "Системный подход получил распространение. Ключевые работы 1970 "
        "и 1990 годов задали основу теории."
    ]
    out, _ = validate_answer(answer, corpus)
    assert "1970-1990" in out


def test_english_1970s_is_stripped_when_ungrounded() -> None:
    answer = "The method was proposed in the 1970s."
    corpus = ["The method is a way to structure an investigation."]
    out, report = validate_answer(answer, corpus)
    assert "1970s" not in out
    assert report.decades_stripped
