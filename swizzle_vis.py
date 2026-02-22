#!/usr/bin/env python3
"""
Single-file HTML tensor visualizer server.

- Backend: Python stdlib `http.server`
  - GET /      -> HTML UI
  - GET /data  -> JSON { data, shape, dtype, highlights }

Integrate your swizzle code by providing `get_swizzle_value(*shape, m, b, s)`
below (copy from your notebook), or import it if you have a module version.
"""

import json
import os
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
import numpy as np

USAGE = """
identity swizzle: B=0, M=4, S=3 (compiler 定义的)

偏移函数
def swizzle_int(ptr_int, b: int, m: int, s: int):
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << (m + s)
    return ptr_int ^ ((ptr_int & yyy_msk) >> s)

"""


# 32 colors (for up to 32 banks).
# bank[0] is intentionally yellow.
COLOR_LIST: list[str] = [
    "gold",  # bank[0]
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]


def custom_cell_color(
    value: int,
    index: list[int],
    *,
    shape: list[int],
    width: int,
    bank_mode: str,
    marked_banks: Optional[list[int]],
) -> Optional[str]:
    """
    Return a CSS color string (e.g. "#ff00aa", "rgba(255,0,0,.4)", "gold"),
    or None for no highlight.
    """
    if bank_mode == "off" or not marked_banks:
        return None

    # 32 banks, 4 bytes per bank.
    if bank_mode == "origin":
        bank_id = (int(value) * int(width) // 4) % 32
    elif bank_mode == "cur":
        bank_id = (_linear_offset(shape, index) * int(width) // 4) % 32
    else:
        raise ValueError(f"unknown bank_mode: {bank_mode!r}")

    for i, b in enumerate(marked_banks):
        if bank_id == int(b):
            return (
                COLOR_LIST[i]
                if i < len(COLOR_LIST)
                else COLOR_LIST[i % len(COLOR_LIST)]
            )
    return None


@cute.jit
def get_swizzle_value_kernel(
    input: cute.Tensor,
    out: cute.Tensor,
    m: cutlass.Constexpr,
    b: cutlass.Constexpr,
    s: cutlass.Constexpr,
):
    sw = cute.make_swizzle(b, m, s)
    composed_layout = cute.make_composed_layout(inner=sw, offset=0, outer=input.layout)
    sw_view = cute.make_tensor(input.iterator, composed_layout)
    for i in range(cute.size(out)):
        out[i] = sw_view[i]


def get_swizzle_value(*shape, m, b, s):
    assert all([x >= 0 for x in [m, b, s]]), "m, b, s should >= 0"
    a = torch.arange(np.prod(shape), dtype=torch.int32).reshape(*shape)
    out = torch.zeros_like(a)
    a_cute = cute.runtime.from_dlpack(a)
    out_cute = cute.runtime.from_dlpack(out)
    get_swizzle_value_kernel(a_cute, out_cute, m, b, s)
    return out


# ---------------------------
# Backend implementation
# ---------------------------
def _parse_shape(shape_text: str) -> list[int]:
    parts = [p.strip() for p in (shape_text or "").split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("shape is empty (example: 32, 2, 32)")
    shape = [int(p) for p in parts]
    if any(d <= 0 for d in shape):
        raise ValueError("shape dimensions must be positive integers")
    return shape


def _parse_int(
    q: dict[str, list[str]], key: str, default: int, *, lo: int, hi: int
) -> int:
    raw = (q.get(key) or [str(default)])[0]
    try:
        v = int(raw)
    except Exception as e:
        print(f"error: {traceback.format_exc()}")
        raise ValueError(f"{key} must be int, got {raw!r}") from e
    if v < lo or v > hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}], got {v}")
    return v


def _parse_optional_int(q: dict[str, list[str]], key: str) -> Optional[int]:
    raw = (q.get(key) or [""])[0].strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except Exception as e:
        raise ValueError(f"{key} must be int, got {raw!r}") from e


def _parse_index(q: dict[str, list[str]]) -> list[int]:
    raw = (q.get("idx") or q.get("index") or [""])[0].strip()
    if raw == "":
        raise ValueError("idx is required (e.g. idx=1,1)")
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p != ""]
    try:
        return [int(p) for p in parts]
    except Exception as e:
        raise ValueError(f"idx must be comma-separated ints, got {raw!r}") from e


def _parse_bank_mode(q: dict[str, list[str]]) -> str:
    raw = (q.get("bank_mode") or [""])[0].strip().lower()
    if raw in ("off", "origin", "cur"):
        return raw

    # Backward compatible: old checkbox `bank_enabled`/`rank_enabled` implies origin mode
    enabled = (q.get("bank_enabled") or q.get("rank_enabled") or ["0"])[0] in (
        "1",
        "true",
        "True",
        "on",
    )
    return "origin" if enabled else "off"


def _parse_bank_list_text(raw: str) -> list[int]:
    raw = (raw or "").strip()
    if raw == "":
        return []
    tokens = [t.strip() for t in raw.split(",")]
    tokens = [t for t in tokens if t != ""]

    out: list[int] = []
    seen: set[int] = set()

    def add(v: int) -> None:
        if v < 0 or v > 31:
            raise ValueError(f"bank id out of range [0,31]: {v}")
        if v in seen:
            return
        seen.add(v)
        out.append(v)

    for tok in tokens:
        if "-" in tok:
            a_str, b_str = [p.strip() for p in tok.split("-", 1)]
            if a_str == "" or b_str == "":
                raise ValueError(f"bad range token: {tok!r}")
            a, b = int(a_str), int(b_str)
            if a > b:
                raise ValueError(f"bad range (start > end): {tok!r}")
            for v in range(a, b + 1):
                add(v)
        else:
            add(int(tok))

    return out


def _parse_marked_banks(q: dict[str, list[str]], bank_mode: str) -> Optional[list[int]]:
    if bank_mode == "off":
        return None
    raw = (q.get("bank") or q.get("rank") or [""])[0].strip()
    try:
        return _parse_bank_list_text(raw)
    except Exception as e:
        raise ValueError(
            f"bank must be ints/ranges like '0-3,4,7-9', got {raw!r}"
        ) from e


def _linear_offset(shape: list[int], index: list[int]) -> int:
    if len(index) != len(shape):
        raise ValueError(f"idx rank mismatch: got {len(index)}, expected {len(shape)}")
    for i, (v, d) in enumerate(zip(index, shape)):
        if v < 0 or v >= d:
            raise ValueError(f"idx[{i}] out of range: {v} not in [0, {d})")
    stride = 1
    off = 0
    for v, d in zip(reversed(index), reversed(shape)):
        off += v * stride
        stride *= d
    return int(off)


def _bin_str(x: int, nbits: int) -> str:
    return format(int(x) & ((1 << nbits) - 1), f"0{nbits}b")


def _log2_width(width: int) -> int:
    w = int(width)
    if w <= 0 or (w & (w - 1)) != 0:
        raise ValueError("width must be power-of-two bytes (1,2,4,8)")
    return w.bit_length() - 1


def _swizzle_explain(
    *,
    shape: list[int],
    index: list[int],
    width: int,
    b: int,
    m: int,
    s: int,
    src_value: Optional[int],
) -> dict[str, Any]:
    dst_off = _linear_offset(shape, index)
    wbits = _log2_width(width)
    # Address integer in bytes (low wbits bits are the element-width alignment).
    # Per UI requirement: compute ptr_int from clicked `value`, not from index-derived offset.
    src = int(dst_off if src_value is None else src_value)
    # origin bank: computed from clicked value (interpreted as element offset)
    origin_bank = (int(src) * int(width) // 4) % 32
    # cur bank: computed from coordinate-derived address (dst_off * width bytes)
    cur_bank = (int(dst_off) * int(width) // 4) % 32
    ptr_int = src << wbits
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << (wbits + m + s)
    and_shift = (ptr_int & yyy_msk) >> s
    out = ptr_int ^ and_shift

    nbits = max(12, (wbits + m + s + b + 1))
    segments = {
        "W": [0, (wbits - 1) if wbits > 0 else -1],
        "S": [wbits, (wbits + s - 1) if s > 0 else -1],
        "M": [wbits + s, (wbits + s + m - 1) if m > 0 else -1],
        "B": [wbits + s + m, (wbits + s + m + b - 1) if b > 0 else -1],
    }
    return {
        "shape": shape,
        "index": index,
        "width": width,
        "wbits": wbits,
        "b": b,
        "m": m,
        "s": s,
        "nbits": nbits,
        "segments": segments,
        "src_value": src_value,
        "dst_off": int(dst_off),
        "origin_bank": int(origin_bank),
        "cur_bank": int(cur_bank),
        "ptr_int": ptr_int,
        "yyy_msk": int(yyy_msk),
        "and_shift": int(and_shift),
        "out": int(out),
        "ptr_bin": _bin_str(ptr_int, nbits),
        "yyy_msk_bin": _bin_str(yyy_msk, nbits),
        "and_shift_bin": _bin_str(and_shift, nbits),
        "out_bin": _bin_str(out, nbits),
    }


def _compute_highlights(
    t, *, width: int, bank_mode: str, marked_banks: Optional[list[int]]
) -> dict[str, str]:
    # Keeps payload compact by only sending colored cells.
    import numpy as np
    import torch

    t_cpu = t.detach()
    if t_cpu.is_cuda:
        t_cpu = t_cpu.cpu()
    if t_cpu.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint8):
        # still allow; we cast per-element below
        pass

    arr = t_cpu.numpy()
    highlights: dict[str, str] = {}
    for idx in np.ndindex(*arr.shape):
        v = int(arr[idx])
        color = custom_cell_color(
            v,
            list(idx),
            shape=list(arr.shape),
            width=width,
            bank_mode=bank_mode,
            marked_banks=marked_banks,
        )
        if color:
            highlights[",".join(map(str, idx))] = color
    return highlights


@dataclass(frozen=True)
class TensorResponse:
    data: Any
    shape: list[int]
    dtype: str
    highlights: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "data": self.data,
                "shape": self.shape,
                "dtype": self.dtype,
                "highlights": self.highlights,
            },
            ensure_ascii=False,
        )


HTML_PAGE = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Torch Tensor Visualizer</title>
    <style>
      :root {
        --bg: #0b0f19;
        --panel: #121a2a;
        --panel2: #0f1626;
        --text: #e7eefc;
        --muted: #9db0d0;
        --border: rgba(231, 238, 252, 0.14);
        --border2: rgba(231, 238, 252, 0.08);
        --accent: #6ea8fe;
        --bad: #ff6b6b;
        --cell: #0c1322;
      }

      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        background: radial-gradient(1200px 700px at 20% -20%, rgba(110,168,254,0.25), transparent 60%),
                    radial-gradient(900px 600px at 120% 0%, rgba(130,216,255,0.16), transparent 55%),
                    var(--bg);
        color: var(--text);
      }

      .wrap {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px 16px 28px;
      }

      .top {
        display: grid;
        gap: 12px;
        grid-template-columns: 1fr;
        align-items: start;
      }

      @media (min-width: 900px) {
        .top { grid-template-columns: 420px 1fr; }
      }

      .card {
        background: linear-gradient(180deg, rgba(18,26,42,0.92), rgba(15,22,38,0.92));
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 20px 70px rgba(0,0,0,0.35);
      }

      .card h2 {
        font-size: 14px;
        font-weight: 650;
        letter-spacing: .2px;
        margin: 0;
        padding: 14px 14px 10px;
        color: var(--text);
        border-bottom: 1px solid var(--border2);
      }

      .controls {
        padding: 12px 14px 14px;
        display: grid;
        gap: 12px;
      }

      .row {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
      }

      .label {
        color: var(--muted);
        font-size: 12px;
        display: flex;
        gap: 10px;
        align-items: baseline;
        justify-content: space-between;
      }

      input[type="text"], input[type="number"], select {
        width: 100%;
        box-sizing: border-box;
        background: rgba(11, 15, 25, 0.65);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 10px;
        outline: none;
      }

      input[type="text"]:focus, input[type="number"]:focus, select:focus {
        border-color: rgba(110,168,254,0.55);
        box-shadow: 0 0 0 3px rgba(110,168,254,0.16);
      }

      .rangeRow {
        display: grid;
        grid-template-columns: 56px 1fr 70px;
        gap: 10px;
        align-items: center;
      }

      input[type="range"] {
        width: 100%;
      }

      .mini {
        font-size: 12px;
        color: var(--muted);
      }

      .rankRow {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 10px;
        align-items: center;
      }

      .rankRow input[type="checkbox"] {
        transform: translateY(1px);
      }

      .status {
        padding: 10px 14px;
        display: flex;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        border-top: 1px solid var(--border2);
        color: var(--muted);
        font-size: 12px;
      }

      .status .err {
        color: var(--bad);
        white-space: pre-wrap;
      }

      .msgblock {
        margin: 0 14px 14px;
        background: rgba(11,15,25,0.55);
        border: 1px solid var(--border2);
        border-radius: 12px;
        overflow: auto;
        max-height: 180px;
      }
      .usageblock {
        max-height: 360px;
      }
      .msgtitle {
        padding: 8px 10px;
        border-bottom: 1px solid var(--border2);
        font-size: 12px;
        color: var(--muted);
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      pre.msgpre {
        margin: 0;
        padding: 10px;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
        color: var(--text);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }

      .swzPop {
        position: fixed;
        z-index: 9999;
        max-width: min(820px, calc(100vw - 24px));
        max-height: min(70vh, 520px);
        overflow: auto;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(18,26,42,0.96), rgba(15,22,38,0.96));
        box-shadow: 0 30px 90px rgba(0,0,0,0.5);
        padding: 12px 12px;
        display: none;
      }
      .swzTitle {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
      }
      .swzTitle .t {
        font-size: 13px;
        font-weight: 650;
      }
      .swzClose {
        border: 1px solid rgba(231,238,252,0.18);
        background: rgba(231,238,252,0.08);
        border-radius: 10px;
        padding: 6px 10px;
      }
      .swzGrid {
        display: grid;
        gap: 10px;
      }
      .swzLine {
        display: grid;
        grid-template-columns: 190px 1fr;
        gap: 10px;
        align-items: baseline;
      }
      .swzKey {
        color: var(--muted);
        font-size: 12px;
      }
      .bits {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        white-space: pre;
      }
      .bit {
        display: inline-block;
        padding: 1px 0;
      }
      .segW { border-bottom: 2px solid rgba(199,125,255,0.9); }
      .segS { border-bottom: 2px solid rgba(110,168,254,0.85); }
      .segM { border-bottom: 2px solid rgba(6,214,160,0.85); }
      .segB { border-bottom: 2px solid rgba(255,209,102,0.95); }
      .swzLegend {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 6px;
        color: var(--muted);
        font-size: 12px;
      }
      .swzLegend span { display: inline-flex; gap: 6px; align-items: center; }
      .dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }

      .viz {
        padding: 14px;
      }

      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px 14px;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
      }

      .pill {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        color: var(--muted);
        background: rgba(11,15,25,0.55);
        border: 1px solid var(--border2);
        border-radius: 999px;
        padding: 6px 10px;
      }

      .slice {
        border: 1px solid var(--border2);
        border-radius: 12px;
        margin: 10px 0;
        overflow: hidden;
        background: rgba(11,15,25,0.35);
      }

      .sliceHeader {
        padding: 8px 10px;
        border-bottom: 1px solid var(--border2);
        font-size: 12px;
        color: var(--muted);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }

      .gridWrap {
        overflow: auto;
        padding: 10px;
      }

      table.grid {
        border-collapse: separate;
        border-spacing: 2px;
      }

      table.grid td {
        background: var(--cell);
        border: 1px solid rgba(231,238,252,0.07);
        border-radius: 6px;
        padding: 6px 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        white-space: nowrap;
        text-align: right;
        min-width: 42px;
      }
      table.grid th {
        background: rgba(11,15,25,0.35);
        border: 1px solid rgba(231,238,252,0.06);
        border-radius: 6px;
        padding: 6px 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        color: var(--muted);
        white-space: nowrap;
        text-align: right;
        min-width: 42px;
      }

      table.grid td[data-hl="1"] {
        outline: 2px solid rgba(255,255,255,0.12);
        outline-offset: -2px;
      }

      .loading {
        color: var(--muted);
        font-size: 12px;
      }

      button {
        cursor: pointer;
        border: 1px solid rgba(110,168,254,0.35);
        background: rgba(110,168,254,0.16);
        color: var(--text);
        border-radius: 10px;
        padding: 9px 10px;
        font-weight: 600;
      }
      button:hover {
        background: rgba(110,168,254,0.22);
      }
      .btnRow {
        display: flex;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">
        <div class="card">
          <h2>Controls</h2>
          <div class="controls">
            <div class="row">
              <div class="label">
                <span>shape</span>
                <span class="mini">comma-separated ints (e.g. 32, 2, 32)</span>
              </div>
              <input id="shape" type="text" spellcheck="false" value="32,32" />
            </div>

            <div class="row">
              <div class="label"><span>B</span><span class="mini">[0, 5]</span></div>
              <div class="rangeRow">
                <input id="b_num" type="number" min="0" max="5" step="1" value="3" />
                <input id="b_rng" type="range" min="0" max="5" step="1" value="3" />
                <div class="mini">b</div>
              </div>
            </div>

            <div class="row">
              <div class="label"><span>M</span><span class="mini">[0, 5]</span></div>
              <div class="rangeRow">
                <input id="m_num" type="number" min="0" max="5" step="1" value="3" />
                <input id="m_rng" type="range" min="0" max="5" step="1" value="3" />
                <div class="mini">m</div>
              </div>
            </div>

            <div class="row">
              <div class="label"><span>S</span><span class="mini">[0, 5]</span></div>
              <div class="rangeRow">
                <input id="s_num" type="number" min="0" max="5" step="1" value="3" />
                <input id="s_rng" type="range" min="0" max="5" step="1" value="3" />
                <div class="mini">s</div>
              </div>
            </div>

            <div class="row">
              <div class="label"><span>width (bytes)</span><span class="mini">passed to color function</span></div>
              <select id="width">
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="4" selected>4</option>
                <option value="8">8</option>
              </select>
            </div>

            <div class="row">
              <div class="label"><span>标记bank</span><span class="mini">comma-separated ints (0-31)</span></div>
              <div class="rankRow">
                <select id="bank_mode">
                  <option value="off" selected>关闭</option>
                  <option value="origin">标记原始</option>
                  <option value="cur">标记目前</option>
                </select>
                <input id="bank" type="text" spellcheck="false" placeholder='e.g. "0,1,2"' disabled />
              </div>
            </div>

            <div class="row">
              <div class="label"><span>show swizzle compute</span><span class="mini">click a cell to explain</span></div>
              <div class="rankRow">
                <label class="mini"><input id="show_swizzle" type="checkbox" checked /> enable</label>
                <div class="mini">When enabled, click a tensor value.</div>
              </div>
            </div>

            <div class="btnRow">
              <button id="refresh">Refresh</button>
              <div id="loading" class="loading" style="display:none;">Loading…</div>
            </div>
          </div>
          <div class="status">
            <div id="ok" class="mini">Ready</div>
          </div>
          <div class="msgblock">
            <div class="msgtitle">
              <div>Messages</div>
              <div class="mini" id="msg_meta"></div>
            </div>
            <pre id="msg" class="msgpre"></pre>
          </div>
          <div class="msgblock usageblock">
            <div class="msgtitle">
              <div>Usage</div>
              <div class="mini">read-only</div>
            </div>
            <pre id="usage" class="msgpre"></pre>
          </div>
        </div>

        <div class="card">
          <h2>Tensor</h2>
          <div class="viz">
            <div class="meta">
              <div class="pill" id="meta_shape">shape: -</div>
              <div class="pill" id="meta_dtype">dtype: -</div>
              <div class="pill" id="meta_hint">Tip: scroll inside grids</div>
            </div>
            <div id="root"></div>
          </div>
        </div>
      </div>
    </div>

    <div id="swz" class="swzPop" role="dialog" aria-modal="false">
      <div class="swzTitle">
        <div class="t" id="swz_title">Swizzle compute</div>
        <button class="swzClose" id="swz_close">Close</button>
      </div>
      <div id="swz_body"></div>
    </div>

    <script>
      const $ = (id) => document.getElementById(id);
      const USAGE_TEXT = __USAGE_JSON__;

      function clampInt(v, lo, hi, fallback) {
        const n = Number.parseInt(v, 10);
        if (Number.isNaN(n)) return fallback;
        return Math.max(lo, Math.min(hi, n));
      }

      function setupSync(name) {
        const num = $(`${name}_num`);
        const rng = $(`${name}_rng`);
        const syncFromNum = () => {
          const v = clampInt(num.value, 0, 5, Number(rng.value));
          num.value = String(v);
          rng.value = String(v);
          scheduleUpdate();
        };
        const syncFromRng = () => {
          num.value = rng.value;
          scheduleUpdate();
        };
        num.addEventListener("input", syncFromNum);
        rng.addEventListener("input", syncFromRng);
      }

      setupSync("b");
      setupSync("m");
      setupSync("s");

      $("shape").addEventListener("input", () => scheduleUpdate());
      $("width").addEventListener("change", () => scheduleUpdate());
      $("refresh").addEventListener("click", () => scheduleUpdate(true));

      $("bank_mode").addEventListener("change", () => {
        const mode = $("bank_mode").value;
        const en = (mode !== "off");
        $("bank").disabled = !en;
        $("bank").style.opacity = en ? "1" : "0.55";
        scheduleUpdate();
      });
      $("bank").addEventListener("input", () => scheduleUpdate());

      let lastShape = null;

      let timer = null;
      let inflight = null;
      let lastOk = "Ready";
      let lastErr = "";

      $("usage").textContent = USAGE_TEXT || "";

      function renderMsg() {
        const parts = [];
        if (lastOk) parts.push(`[status] ${lastOk}`);
        if (lastErr) parts.push(`[error]\n${lastErr}`);
        $("msg").textContent = parts.join("\n\n");
        $("msg_meta").textContent = `${new Date().toLocaleTimeString()}`;
      }

      function setError(msg) {
        lastErr = msg || "";
        renderMsg();
      }
      function setOk(msg) {
        lastOk = msg || "";
        $("ok").textContent = lastOk;
        renderMsg();
      }
      function setLoading(on) {
        $("loading").style.display = on ? "block" : "none";
      }

      function scheduleUpdate(immediate=false) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(update, immediate ? 0 : 160);
      }

      function buildQuery() {
        const params = new URLSearchParams();
        params.set("shape", $("shape").value);
        params.set("b", $("b_num").value);
        params.set("m", $("m_num").value);
        params.set("s", $("s_num").value);
        params.set("width", $("width").value);
        params.set("bank_mode", $("bank_mode").value);
        params.set("bank", $("bank").value);
        return params.toString();
      }

      function isArray(x) { return Array.isArray(x); }
      function isNumber(x) { return typeof x === "number"; }

      function isMatrix2D(node) {
        return isArray(node) && node.length > 0 && isArray(node[0]) && (node[0].length === 0 || isNumber(node[0][0]));
      }

      function renderMatrix2D(mat, prefix, highlights) {
        const table = document.createElement("table");
        table.className = "grid";

        // Column ticks (every 5)
        const thead = document.createElement("thead");
        const hr = document.createElement("tr");
        const corner = document.createElement("th");
        corner.textContent = "";
        hr.appendChild(corner);
        const cols = mat.length > 0 ? mat[0].length : 0;
        for (let c = 0; c < cols; c++) {
          const th = document.createElement("th");
          th.textContent = (c % 5 === 0) ? String(c) : "";
          hr.appendChild(th);
        }
        thead.appendChild(hr);
        table.appendChild(thead);

        const tbody = document.createElement("tbody");
        for (let r = 0; r < mat.length; r++) {
          const tr = document.createElement("tr");
          // Row ticks (every 5)
          const rh = document.createElement("th");
          rh.textContent = (r % 5 === 0) ? String(r) : "";
          tr.appendChild(rh);
          const row = mat[r];
          for (let c = 0; c < row.length; c++) {
            const td = document.createElement("td");
            const idx = prefix.concat([r, c]).join(",");
            const v = row[c];
            td.textContent = String(v);
            td.dataset.idx = idx;
            const color = highlights[idx];
            if (color) {
              td.style.background = color;
              td.dataset.hl = "1";
            }
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        return table;
      }

      function renderNode(node, prefix, highlights) {
        if (isMatrix2D(node)) {
          const wrap = document.createElement("div");
          wrap.className = "gridWrap";
          wrap.appendChild(renderMatrix2D(node, prefix, highlights));
          return wrap;
        }

        if (isArray(node) && node.length > 0 && isNumber(node[0])) {
          // 1D: render as a 1-row matrix
          return renderNode([node], prefix, highlights);
        }

        if (isArray(node)) {
          const container = document.createElement("div");
          for (let i = 0; i < node.length; i++) {
            const slice = document.createElement("div");
            slice.className = "slice";

            const hdr = document.createElement("div");
            hdr.className = "sliceHeader";
            const left = document.createElement("div");
            left.textContent = `slice @ [${prefix.concat([i]).join(", ")}]`;
            const right = document.createElement("div");
            right.className = "mini";
            right.textContent = "";
            hdr.appendChild(left);
            hdr.appendChild(right);

            const body = renderNode(node[i], prefix.concat([i]), highlights);
            slice.appendChild(hdr);
            slice.appendChild(body);
            container.appendChild(slice);
          }
          return container;
        }

        // scalar fallback
        const div = document.createElement("div");
        div.textContent = String(node);
        return div;
      }

      async function update() {
        if (inflight) inflight.abort();
        inflight = new AbortController();
        setError("");
        setLoading(true);
        setOk("Fetching…");

        try {
          const qs = buildQuery();
          const res = await fetch(`/data?${qs}`, { signal: inflight.signal });
          const txt = await res.text();
          let payload = null;
          try { payload = JSON.parse(txt); } catch(e) { throw new Error(`Bad JSON: ${txt.slice(0, 200)}`); }
          if (!res.ok) {
            throw new Error(payload && payload.error ? payload.error : `HTTP ${res.status}`);
          }

          $("meta_shape").textContent = `shape: [${payload.shape.join(", ")}]`;
          $("meta_dtype").textContent = `dtype: ${payload.dtype}`;
          lastShape = payload.shape;

          const root = $("root");
          root.innerHTML = "";
          const highlights = payload.highlights || {};
          root.appendChild(renderNode(payload.data, [], highlights));

          setOk("OK");
        } catch (e) {
          if (e.name === "AbortError") return;
          setError(String(e.message || e));
          setOk("Error");
        } finally {
          setLoading(false);
        }
      }

      function bitsHtml(binStr, segments) {
        const n = binStr.length;
        const clsForBitPos = (pos) => {
          // pos is LSB=0
          if (segments && segments.W && pos >= segments.W[0] && pos <= segments.W[1]) return "segW";
          if (segments && segments.S && pos >= segments.S[0] && pos <= segments.S[1]) return "segS";
          if (segments && segments.M && pos >= segments.M[0] && pos <= segments.M[1]) return "segM";
          if (segments && segments.B && pos >= segments.B[0] && pos <= segments.B[1]) return "segB";
          return "";
        };
        const parts = [];
        parts.push('<span class="bits">0b');
        for (let i = 0; i < n; i++) {
          const ch = binStr[i];
          const bitPos = (n - 1 - i);
          const cls = clsForBitPos(bitPos);
          parts.push(`<span class="bit ${cls}">${ch}</span>`);
          const fromLeft = i + 1;
          if (fromLeft % 4 === 0 && fromLeft !== n) parts.push(" ");
        }
        parts.push("</span>");
        return parts.join("");
      }

      function showSwzAt(x, y, title, html) {
        const pop = $("swz");
        $("swz_title").textContent = title;
        $("swz_body").innerHTML = html;
        pop.style.display = "block";
        const pad = 12;
        const rect = pop.getBoundingClientRect();
        let left = x + 12;
        let top = y + 12;
        if (left + rect.width + pad > window.innerWidth) left = Math.max(pad, window.innerWidth - rect.width - pad);
        if (top + rect.height + pad > window.innerHeight) top = Math.max(pad, window.innerHeight - rect.height - pad);
        pop.style.left = `${left}px`;
        pop.style.top = `${top}px`;
      }

      function hideSwz() {
        $("swz").style.display = "none";
      }

      $("swz_close").addEventListener("click", hideSwz);
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideSwz(); });
      document.addEventListener("click", (e) => {
        const pop = $("swz");
        if (pop.style.display !== "block") return;
        if (pop.contains(e.target)) return;
        // don't auto-close when clicking a td (we'll update it)
        if (e.target && e.target.tagName === "TD") return;
        hideSwz();
      });

      $("root").addEventListener("click", async (e) => {
        if (!$("show_swizzle").checked) return;
        const td = e.target && e.target.tagName === "TD" ? e.target : null;
        if (!td) return;
        const idx = td.dataset.idx;
        if (!idx) return;
        if (!lastShape) return;
        try {
          const params = new URLSearchParams();
          params.set("shape", $("shape").value);
          params.set("idx", idx);
          params.set("value", td.textContent);
          params.set("b", $("b_num").value);
          params.set("m", $("m_num").value);
          params.set("s", $("s_num").value);
          params.set("width", $("width").value);
          const res = await fetch(`/swizzle?${params.toString()}`);
          const txt = await res.text();
          let payload = null;
          try { payload = JSON.parse(txt); } catch(e2) { throw new Error(`Bad JSON: ${txt.slice(0, 200)}`); }
          if (!res.ok) throw new Error(payload && payload.error ? payload.error : `HTTP ${res.status}`);

          const seg = payload.segments || {};
          const legend = `
            <div class="swzLegend">
              <span><i class="dot" style="background: rgba(199,125,255,0.9)"></i>W (log2(width)=${payload.wbits})</span>
              <span><i class="dot" style="background: rgba(110,168,254,0.85)"></i>S</span>
              <span><i class="dot" style="background: rgba(6,214,160,0.85)"></i>M</span>
              <span><i class="dot" style="background: rgba(255,209,102,0.95)"></i>B</span>
            </div>`;

          const body = `
            <div class="swzGrid">
              <div class="swzLine"><div class="swzKey">ptr_int</div><div>${bitsHtml(payload.ptr_bin, seg)}</div></div>
              <div class="swzLine"><div class="swzKey">yyy_msk</div><div>${bitsHtml(payload.yyy_msk_bin, seg)}</div></div>
              <div class="swzLine"><div class="swzKey">(ptr_int & yyy_msk) >> s</div><div>${bitsHtml(payload.and_shift_bin, seg)}</div></div>
              <div class="swzLine"><div class="swzKey">ptr_int ^ ((...) >> s)</div><div>${bitsHtml(payload.out_bin, seg)}</div></div>
              <div class="swzLine"><div class="swzKey">After Swizzle</div><div>${payload.src_value} -> ${payload.dst_off} (${payload.index.join(",")})</div></div>
              <div class="swzLine"><div class="swzKey">bank</div><div>${payload.origin_bank} -> ${payload.cur_bank}</div></div>
              ${legend}
            </div>`;

          showSwzAt(e.clientX, e.clientY, `Swizzle compute @ [${idx}] (b=${payload.b}, m=${payload.m}, s=${payload.s})`, body);
        } catch (err) {
          showSwzAt(e.clientX, e.clientY, `Swizzle compute @ [${idx}]`, `<pre class="msgpre">${String(err.message || err)}</pre>`);
        }
      });

      // initial
      scheduleUpdate(true);
    </script>
  </body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "TensorVizHTTP/0.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = HTML_PAGE.replace("__USAGE_JSON__", json.dumps(USAGE))
                self._send(
                    HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8"
                )
                return

            if parsed.path == "/data":
                q = parse_qs(parsed.query)
                shape = _parse_shape((q.get("shape") or [""])[0])
                b = _parse_int(q, "b", 3, lo=0, hi=5)
                m = _parse_int(q, "m", 3, lo=0, hi=5)
                s = _parse_int(q, "s", 3, lo=0, hi=5)
                width = _parse_int(q, "width", 4, lo=1, hi=8)
                if width not in (1, 2, 4, 8):
                    raise ValueError("width must be one of 1,2,4,8")
                bank_mode = _parse_bank_mode(q)
                marked_banks = _parse_marked_banks(q, bank_mode)

                t = get_swizzle_value(*shape, m=m, b=b, s=s)
                t_cpu = t.detach()
                if getattr(t_cpu, "is_cuda", False):
                    t_cpu = t_cpu.cpu()

                resp = TensorResponse(
                    data=t_cpu.tolist(),
                    shape=shape,
                    dtype=str(getattr(t_cpu, "dtype", "unknown")),
                    highlights=_compute_highlights(
                        t, width=width, bank_mode=bank_mode, marked_banks=marked_banks
                    ),
                )
                self._send(
                    HTTPStatus.OK,
                    resp.to_json().encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return

            if parsed.path == "/swizzle":
                q = parse_qs(parsed.query)
                shape = _parse_shape((q.get("shape") or [""])[0])
                index = _parse_index(q)
                src_value = _parse_optional_int(q, "value")
                b = _parse_int(q, "b", 3, lo=0, hi=5)
                m = _parse_int(q, "m", 3, lo=0, hi=5)
                s = _parse_int(q, "s", 3, lo=0, hi=5)
                width = _parse_int(q, "width", 4, lo=1, hi=8)
                if width not in (1, 2, 4, 8):
                    raise ValueError("width must be one of 1,2,4,8")
                resp = _swizzle_explain(
                    shape=shape,
                    index=index,
                    width=width,
                    b=b,
                    m=m,
                    s=s,
                    src_value=src_value,
                )
                self._send(
                    HTTPStatus.OK,
                    json.dumps(resp, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return

            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
        except Exception as e:
            print(f"error: {traceback.format_exc()}")
            err = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=12),
            }
            self._send(
                HTTPStatus.BAD_REQUEST,
                json.dumps(err, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        # quiet by default; set TENSOR_VIZ_LOG=1 to enable
        if os.environ.get("TENSOR_VIZ_LOG") == "1":
            super().log_message(fmt, *args)


def main() -> None:
    host = os.environ.get("TENSOR_VIZ_HOST", "0.0.0.0")
    port = int(os.environ.get("TENSOR_VIZ_PORT", "8008"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Tensor visualizer running at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
