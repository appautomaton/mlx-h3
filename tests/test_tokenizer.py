"""Tests for the dependency-free Qwen byte-level BPE tokenizer."""

from __future__ import annotations

from mlx_h3 import tokenizer


def tiny_tokenizer() -> tokenizer.QwenTokenizer:
    vocab = {
        symbol: index for index, symbol in enumerate(tokenizer._BYTE_ENCODER.values())
    }
    for symbol in ("he", "hel", "hell", "hello"):
        vocab[symbol] = len(vocab)
    return tokenizer.QwenTokenizer(
        {
            "normalizer": {"type": "NFC"},
            "added_tokens": [
                {
                    "id": 300,
                    "content": "<special>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                }
            ],
            "model": {
                "type": "BPE",
                "unk_token": None,
                "vocab": vocab,
                "merges": ["h e", "he l", "hel l", "hell o"],
            },
        }
    )


def test_unicode_split_matches_the_released_regex_semantics():
    assert tokenizer._pretokenize("Hello, WORLD!  2026\n中文 café's") == [
        "Hello",
        ",",
        " WORLD",
        "!",
        " ",
        " ",
        "2",
        "0",
        "2",
        "6",
        "\n",
        "中文",
        " café",
        "'s",
    ]
    assert tokenizer._pretokenize("a   b") == ["a", "  ", " b"]
    assert tokenizer._pretokenize("a \n  b") == ["a", " \n", " ", " b"]
    assert tokenizer._pretokenize("🙂a!\n") == ["🙂a", "!\n"]


def test_ranked_bpe_added_tokens_and_utf8_round_trip():
    tok = tiny_tokenizer()
    assert tok.encode("hello") == [tok.vocab["hello"]]
    text = "hello café 中文🙂<special>done"
    token_ids = tok.encode(text)
    assert token_ids.count(300) == 1
    assert tok.decode(token_ids) == text


def test_released_tokenizer_uses_raw_prompt_without_special_tokens(local_file):
    path = local_file("MLX_H3_TOKENIZER_FILE")
    tok = tokenizer.QwenTokenizer.from_file(path)
    text = "Tokenizer round-trip: café 中文🙂"
    token_ids = tok.encode(text)
    assert token_ids
    assert token_ids[0] != tokenizer.PAD_TOKEN_ID
    assert tok.decode(token_ids) == text
    assert tok.encode("") == []
    assert tok.encode_prompt("") == [tokenizer.PAD_TOKEN_ID]
    assert tok.encode("<|im_start|>") == [151644]
