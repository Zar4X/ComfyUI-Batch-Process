# ComfyUI Batch Process

A collection of ComfyUI nodes for batch processing text, images, videos, LoRAs, masks, and 3D models.

## Core Features

### Text Processing

- Load TXT files sequentially or randomly
- Filter by filename, prefix, suffix, or file contents
- Select a 1-based file range
- Modify complete strings, prefixes, or suffixes
- Use the selected TXT filename in save filename templates

### Image Processing

- Load images sequentially or randomly
- Filter by filename and optionally scan subfolders
- Return one image, a combined image batch, or an image list
- Center-pad different resolutions when building a batch
- Save PNG, JPG, WebP, BMP, TIFF, and GIF files
- Save matching TXT captions and append animation frames

### Video Processing

- Save ComfyUI `VIDEO` objects in supported containers and codecs
- Preserve available source FPS
- Save matching TXT captions and workflow metadata

### LoRA Processing

- Load LoRAs sequentially or randomly
- Filter by filename, prefix, or suffix
- Apply separate model and CLIP strengths

### General, Mask, and 3D Processing

- Group compatible images, latents, values, `MESH`, and `File3D` objects
- Filter out stale asset slots left over from previous runs
- Select, repeat, copy, and combine mask batches
- Save and preview batches of `MESH`, `File3D`, and supported 3D files
- Save optional TXT captions next to 3D models

## Installation

Clone the repository into the ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Zar4X/ComfyUI-Batch-Process.git batch-process
pip install -r batch-process/requirements.txt
```

Restart ComfyUI after installing or updating the package.

## Nodes Overview

### TXT Batch Loader

Loads UTF-8 TXT files with filename/content filtering, natural sorting, index ranges, incremental loading, seeded random selection, and reset control.

**Outputs:** text content and filename without the extension.

### Text Modify Tool

Searches, replaces, or removes complete text, filename prefixes, or filename suffixes using a custom or automatically detected delimiter.

### Image Batch Loader

Loads PNG, JPG/JPEG, WebP, BMP, and GIF files with filtering, natural sorting, recursive search, index ranges, and sequential/random modes.

**Outputs:** selected image, filename, image count, combined image batch, and image list.

Images with different resolutions are center-padded with black borders for the combined batch. The image list retains individual dimensions.

### Image Batch Saver

Saves images and optional TXT captions with configurable output paths, prefixes, suffixes, delimiters, counters, formats, workflow metadata, frame ranges, and animation frame appending.

### Video Batch Saver

Saves ComfyUI video objects and optional TXT captions with configurable paths, naming, containers, codecs, counters, and workflow metadata.

### LoRA Batch Loader

Loads `.safetensors`, `.ckpt`, `.pt`, and `.bin` LoRAs with filename filtering, incremental/random modes, and separate model/CLIP strengths.

**Outputs:** patched model, patched CLIP, and selected filename.

### Any Batch Group

Combines compatible values into a batch. Supports image tensors, tensor lists, latents, strings, numbers, `MESH`, and `File3D` objects. Different image sizes are center-padded automatically.

### Asset Filter

Removes stale asset slots that ComfyUI keeps from previous runs. Connect reference images (`IMAGE`) and videos (any payload, e.g. `VHS_LoadVideo` output) — the input list grows automatically when the last free slot is connected, with image slots grouped above video slots. Each zone stops growing once it has one input per matching output (9 images, 3 videos), since an input past the last output would have nowhere to go.

Each slot is traced upstream to its source filename. A slot passes when its filename changed since this node last ran; unchanged slots are filtered out. Special cases: the very first run (no saved state) falls back to file modification time within `freshness_hours` (default 24), a re-run where nothing changed reuses the previous run's selection, and slots that are not file-backed (produced by the current run) always pass. If every connected asset would be filtered, all of them pass through as a fallback and a warning is printed.

**Outputs:** `image_0`..`image_8`, `video_0`..`video_2`, plus `image_count` and `video_count`. Slot indices are 0-based so they line up one-to-one with consumers like MiniMax H3 (`image_0` -> `ref_image_0`, and so on).

Outputs mirror the inputs positionally: input `image_N` always leaves on output `image_N`, so a slot's identity never shifts and a preview wired to one slot cannot start showing another slot's image. Each output carries a single asset at its original resolution — nothing is batched, so no black-border padding is ever added. Outputs whose input was filtered out (or is unconnected) return `None`.

Wire `image_0`..`image_8` into a reference consumer's fixed slots once (for example MiniMax H3's `ref_image_0..8` / `ref_video_0..2`) and leave them wired. Such consumers skip `None` slots, so the consumer runs exactly once with only the surviving assets no matter how many survive, and gaps in the middle are harmless.

`None` will, however, crash consumers that do not tolerate it — `PreviewImage` and `SaveImage` among them. Feed those from a slot you know is populated, or gate them on `image_count`.

### 3D Model Batch Saver

Saves `MESH`, `File3D`, supported file paths, and lists of 3D models. `MESH` is written as GLB; file-based models retain their original format.

### 3D Model Batch Preview

Creates ComfyUI 3D previews for individual models or model batches.

### Mask Nodes

- **Mask From Batch:** selects a range from a mask batch
- **Mask Repeat Batch:** repeats masks along the batch dimension
- **Mask Batch Copy:** creates two independent mask copies
- **Mask Batch Composite:** concatenates two mask batches

## Runtime TXT Filename Templates

ComfyUI normally resolves filename placeholders from input widgets only. This package also supports the runtime `filename` output from TXT Batch Loader.

Example Save Image prefix:

```text
D_%TXTBatchLoader.filename%_%EasySeed.seed%
```

For `forest_scene.txt` and seed `197152849854180`, the resolved prefix is:

```text
D_forest_scene_197152849854180
```

The placeholder uses the node's `Node name for S&R` value. Custom names are supported, for example `%Text Source.filename%`. The TXT Batch Loader must participate in the current execution path before the saver runs.

## Common Index Rules

- `start_index = 0`: start from the beginning
- `end_index = 0`: continue to the final item
- Positive start/end values are 1-based; the end is inclusive
- Toggle `reset_on_queue` to restart an incremental loader from its first filtered item

## Requirements

- ComfyUI
- Pillow 10 or newer
- PyTorch 2.0 or newer
- Codecs required by the selected video format

Current package version: `1.2.0`
