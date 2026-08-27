import pytest

from app.services.chunking import chunk_text


def test_short_text_produces_a_single_chunk():
    text = "This is a short document with just a few words in it."
    chunks = chunk_text(text, chunk_words=500, overlap_words=75)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index == 0


def test_long_text_produces_multiple_chunks_with_exact_overlap():
    long_text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(long_text, chunk_words=500, overlap_words=75)
    assert len(chunks) >= 2

    chunk0_words = chunks[0].text.split()
    chunk1_words = chunks[1].text.split()
    # The last 75 words of chunk 0 must be exactly the first 75 words of
    # chunk 1 - that's the whole point of overlap (a fact near a boundary
    # shouldn't be unfindable by either chunk's embedding).
    assert chunk0_words[-75:] == chunk1_words[:75]


def test_chunk_indices_are_sequential():
    long_text = " ".join(f"word{i}" for i in range(1500))
    chunks = chunk_text(long_text, chunk_words=500, overlap_words=75)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_all_words_are_covered_no_gaps():
    long_text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(long_text, chunk_words=500, overlap_words=75)
    # last chunk's last word should be the final word of the source text
    assert chunks[-1].text.split()[-1] == "word1199"
    # first chunk's first word should be the first word of the source text
    assert chunks[0].text.split()[0] == "word0"


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_whitespace_heavy_paragraphs_are_handled():
    text = "Para one.\n\n\n\nPara two with lots of blank lines.\n\nPara three."
    chunks = chunk_text(text, chunk_words=500, overlap_words=75)
    assert len(chunks) == 1
    assert "Para one" in chunks[0].text
    assert "Para three" in chunks[0].text


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_words=100, overlap_words=100)
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_words=100, overlap_words=150)
