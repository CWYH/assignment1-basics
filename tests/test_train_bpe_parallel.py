"""Check document-aligned multiprocessing against sequential pretokenization."""

import pytest

from cs336_basics.train_bpe import _count_pretokens, _document_ranges


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_parallel_counting_matches_sequential(tmp_path, chunk_size):
    special = "<|endoftext|>"
    text = special + special.join(["你好 café!\r\n", "aaa aaa\r", " hi\n\nthere", "😀 café"] * 10) + special
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    expected = _count_pretokens(path, [special], num_workers=1, chunk_size=chunk_size)
    actual = _count_pretokens(path, [special], num_workers=2, chunk_size=chunk_size)
    assert actual == expected
    assert not any(special.encode() in token for token in actual)


def test_range_search_handles_separator_across_read_boundary(tmp_path):
    separator = b"<|endoftext|>"
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"a" * 65535 + separator + b"tail")
    ranges = _document_ranges(path, separator, chunk_size=2)
    assert ranges == [(0, 65535), (65535, path.stat().st_size)]
