# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang python/sglang/srt/function_call/utils.py
# (Apache-2.0). Trimmed to the three helpers the vendored detectors call.
"""Helpers used by the vendored tool-call detectors.

The schema-constraint utilities (``_get_tool_schema``, ``get_json_schema_constraint``,
``infer_type_from_json_schema``) from upstream SGLang are omitted — EdgeVox does
not do constrained decoding, so they'd be dead code. Restore them from upstream
if/when that changes.
"""

from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any

import orjson
import partial_json_parser
from partial_json_parser.core.options import Allow


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> tuple[Any, int]:
    """Parse incomplete JSON commonly encountered during streaming.

    Returns (parsed_object, consumed_length).
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False
