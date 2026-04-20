from __future__ import annotations

from src.rag.chunking import split_pages_into_chunks


def test_small_page_becomes_single_chunk():
    pages = [(1, "Короткий текст на одной странице.")]
    chunks = split_pages_into_chunks(
        pages, subject_slug="s", book="b.pdf", chunk_size=650, overlap=130
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Короткий текст на одной странице."
    assert chunks[0].page == 1
    assert chunks[0].subject_slug == "s"
    assert chunks[0].chunk_idx == 0


def test_all_chunks_within_size_limit():
    sentence = "Это длинное предложение, " + "слово " * 50 + "."
    text = " ".join([sentence] * 5)
    pages = [(1, text)]
    chunks = split_pages_into_chunks(
        pages, subject_slug="s", book="b.pdf", chunk_size=200, overlap=40
    )
    assert all(len(c.text) <= 200 for c in chunks), [len(c.text) for c in chunks]
    assert all(c.chunk_idx == i for i, c in enumerate(chunks))


def test_single_long_sentence_is_hard_split():
    long_sent = "слово " * 400  # ~2400 chars, no sentence boundary
    pages = [(1, long_sent)]
    chunks = split_pages_into_chunks(
        pages, subject_slug="s", book="b.pdf", chunk_size=500, overlap=100
    )
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_page_numbers_preserved():
    pages = [(7, "первая"), (42, "вторая")]
    chunks = split_pages_into_chunks(
        pages, subject_slug="s", book="b.pdf", chunk_size=100, overlap=20
    )
    assert [c.page for c in chunks] == [7, 42]
