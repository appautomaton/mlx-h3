"""Dependency-free byte-level BPE tokenizer used by MiniMax-H3.

The released tokenizer is a Qwen2 byte-level BPE tokenizer.  Pulling the
``tokenizers`` wheel into the runtime also pulls an otherwise unused networking
stack, so this module implements only the exact encode/decode pipeline described
by the local ``tokenizer.json``: NFC normalization, the shipped Unicode split,
GPT-2 byte mapping, and ranked BPE merges.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

PAD_TOKEN_ID = 151643


def _bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    chars = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(value) for value in chars), strict=True))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {value: key for key, value in _BYTE_ENCODER.items()}
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_number(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _pretokenize(text: str) -> list[str]:
    """Match the shipped Qwen Unicode split without a third-party regex engine."""
    pieces: list[str] = []
    index = 0
    while index < len(text):
        tail = text[index:].lower()
        contraction = next(
            (item for item in _CONTRACTIONS if tail.startswith(item)), None
        )
        if contraction is not None:
            pieces.append(text[index : index + len(contraction)])
            index += len(contraction)
            continue

        char = text[index]
        letter_start = index
        if _is_letter(char):
            pass
        elif (
            char not in "\r\n"
            and not _is_number(char)
            and index + 1 < len(text)
            and _is_letter(text[index + 1])
        ):
            letter_start += 1
        else:
            letter_start = -1
        if letter_start >= 0:
            stop = letter_start
            while stop < len(text) and _is_letter(text[stop]):
                stop += 1
            pieces.append(text[index:stop])
            index = stop
            continue

        if _is_number(char):
            pieces.append(char)
            index += 1
            continue

        punct_start = index + int(
            char == " "
            and index + 1 < len(text)
            and not text[index + 1].isspace()
            and not _is_letter(text[index + 1])
            and not _is_number(text[index + 1])
        )
        stop = punct_start
        while (
            stop < len(text)
            and not text[stop].isspace()
            and not _is_letter(text[stop])
            and not _is_number(text[stop])
        ):
            stop += 1
        if stop > punct_start:
            while stop < len(text) and text[stop] in "\r\n":
                stop += 1
            pieces.append(text[index:stop])
            index = stop
            continue

        if char.isspace():
            whitespace_end = index + 1
            while whitespace_end < len(text) and text[whitespace_end].isspace():
                whitespace_end += 1

            newline_end = max(
                (
                    offset + 1
                    for offset in range(index, whitespace_end)
                    if text[offset] in "\r\n"
                ),
                default=-1,
            )
            if newline_end >= 0:
                pieces.append(text[index:newline_end])
                index = newline_end
            elif whitespace_end == len(text):
                pieces.append(text[index:whitespace_end])
                index = whitespace_end
            elif whitespace_end - index > 1:
                pieces.append(text[index : whitespace_end - 1])
                index = whitespace_end - 1
            else:
                pieces.append(char)
                index += 1
            continue

        # Every non-whitespace codepoint not handled above belongs to the
        # punctuation/symbol branch, so reaching here means the tokenizer spec
        # changed in a way this focused implementation does not understand.
        raise ValueError(f"unable to pre-tokenize codepoint U+{ord(char):04X}")

    return pieces


class QwenTokenizer:
    """Qwen byte-level BPE loaded directly from ``tokenizer.json``."""

    def __init__(self, config: dict):
        model = config.get("model", {})
        if model.get("type") != "BPE" or model.get("unk_token") is not None:
            raise ValueError("expected the released no-UNK BPE tokenizer")
        if config.get("normalizer") != {"type": "NFC"}:
            raise ValueError("expected the released NFC tokenizer normalizer")

        self.vocab = {str(key): int(value) for key, value in model["vocab"].items()}
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}
        self.merge_ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            pair = tuple(merge.split(" ", 1)) if isinstance(merge, str) else tuple(merge)
            if len(pair) != 2:
                raise ValueError(f"invalid BPE merge at rank {rank}: {merge!r}")
            self.merge_ranks[(pair[0], pair[1])] = rank

        added = config.get("added_tokens", [])
        if any(
            token.get("single_word")
            or token.get("lstrip")
            or token.get("rstrip")
            or token.get("normalized", True)
            for token in added
        ):
            raise ValueError("unsupported added-token matching policy")
        self.added_tokens = {
            str(token["content"]): int(token["id"]) for token in added
        }
        self.inverse_added_tokens = {
            value: key for key, value in self.added_tokens.items()
        }
        alternatives = sorted(self.added_tokens, key=len, reverse=True)
        self._added_pattern = (
            re.compile("|".join(re.escape(token) for token in alternatives))
            if alternatives
            else None
        )
        self._bpe_cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> QwenTokenizer:
        with Path(path).open(encoding="utf-8") as file:
            return cls(json.load(file))

    def _bpe(self, piece: str) -> tuple[int, ...]:
        encoded = "".join(_BYTE_ENCODER[byte] for byte in piece.encode("utf-8"))
        cached = self._bpe_cache.get(encoded)
        if cached is not None:
            return cached

        symbols = list(encoded)
        while len(symbols) > 1:
            ranked = [
                (self.merge_ranks.get((left, right)), index)
                for index, (left, right) in enumerate(zip(symbols, symbols[1:]))
            ]
            available = [(rank, index) for rank, index in ranked if rank is not None]
            if not available:
                break
            _, best = min(available)
            pair = (symbols[best], symbols[best + 1])
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (
                    symbols[index], symbols[index + 1]
                ) == pair:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged

        try:
            token_ids = tuple(self.vocab[symbol] for symbol in symbols)
        except KeyError as error:
            raise ValueError(f"BPE symbol absent from vocabulary: {error.args[0]!r}") from error
        self._bpe_cache[encoded] = token_ids
        return token_ids

    def _encode_text(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", text)
        return [
            token_id
            for piece in _pretokenize(normalized)
            for token_id in self._bpe(piece)
        ]

    def encode(self, text: str) -> list[int]:
        """Encode raw text without BOS, EOS, padding, or a chat template."""
        if self._added_pattern is None:
            return self._encode_text(text)

        output: list[int] = []
        start = 0
        for match in self._added_pattern.finditer(text):
            output.extend(self._encode_text(text[start : match.start()]))
            output.append(self.added_tokens[match.group()])
            start = match.end()
        output.extend(self._encode_text(text[start:]))
        return output

    def encode_prompt(self, text: str) -> list[int]:
        """MiniMax uses one pad token only when the raw prompt is empty."""
        token_ids = self.encode(text)
        return token_ids or [PAD_TOKEN_ID]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        """Decode IDs, preserving added tokens instead of silently dropping them."""
        parts: list[str] = []
        byte_chars: list[str] = []

        def flush() -> None:
            if not byte_chars:
                return
            raw = bytes(_BYTE_DECODER[char] for char in "".join(byte_chars))
            parts.append(raw.decode("utf-8", errors="replace"))
            byte_chars.clear()

        for token_id in token_ids:
            if token_id in self.inverse_added_tokens:
                flush()
                parts.append(self.inverse_added_tokens[token_id])
            else:
                try:
                    byte_chars.append(self.inverse_vocab[token_id])
                except KeyError as error:
                    raise ValueError(f"unknown token id {token_id}") from error
        flush()
        return "".join(parts)
