## SwizzleVis

[English](README.md) | [中文](README_cn.md)

A single-file HTML tensor / swizzle visualizer (Python `http.server` backend). It provides an interactive UI in your browser for inspecting tensor layouts, index mapping, and optional bank highlighting.

### Quick start

```bash
pip install -r requirements.txt
python swizzle_vis.py
```

Then open `http://localhost:8008` (by default it listens on `0.0.0.0:8008`).

### Optional environment variables

- **`TENSOR_VIZ_HOST`**: bind address, default `0.0.0.0`
- **`TENSOR_VIZ_PORT`**: port number, default `8008`
- **`TENSOR_VIZ_LOG`**: set to `1` to enable request logs (quiet by default)

### Screenshots

#### `assets/show.png`

<img src="assets/show.png" width="1000" />

#### `assets/sw_cal_block.png`

<img src="assets/sw_cal_block.png" width="1000" />

#### `assets/m8n8.png`

<img src="assets/m8n8.png" width="1000" />