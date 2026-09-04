# ComfyUI Batch Process

A collection of ComfyUI nodes for batch processing text, images, videos, LoRAs, masks, and 3D models.

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

Filters out stale asset slots that ComfyUI keeps from previous runs. Each slot is traced upstream to its source filename: changed names pass, unchanged names are dropped. The first run falls back to the `freshness_hours` (default 24) mtime window, a pure re-run reuses the previous selection, and if everything would be dropped all slots pass as a fallback. Inputs grow as you connect them, capped at 9 images / 3 videos, with image slots grouped above video slots.

**Outputs:** `image_0`..`image_8`, `video_0`..`video_2`, `image_count`, `video_count`. Indices are 0-based so they line up one-to-one with MiniMax H3 (`image_0` -> `ref_image_0`). Outputs mirror inputs positionally and each carries a single asset at its original resolution (no batching, no padding); filtered or unconnected slots emit `None`. Wire every output into a reference consumer once and leave it wired — it skips `None` and runs once with the survivors. Consumers that do not tolerate `None` (`PreviewImage`, `SaveImage`) would crash; feed those from a slot you know is populated.

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

The node name may be the class type (`TXTBatchLoader`), the display name, the workflow title, or the node's `Node name for S&R` value — for example `%Text Source.filename%`. The placeholder resolves when the loader runs in the same prompt as the saver; if ComfyUI reuses the loader's cached output instead, the most recently recorded filename is used.

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

Current package version: `1.2.1`
