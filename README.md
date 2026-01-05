# ComfyUI Batch Process
*A collection of nodes for batch processing texts, images, videos, and LoRAs in ComfyUI*

---

## Core Features

### 🔧 Text Processing
- `txt_batch_loader` - Load and filter text files by filename patterns and content
- `text_modify_tool` - Edit text content with search/replace operations on prefixes, suffixes, or whole text

### 🖼️ Image Processing
- `image_batch_loader` - Load images with filtering, subfolder search, index range selection, and list output options
- `image_batch_saver` - Save images with companion text files, frame accumulation, and customizable naming

### 🎬 Video Processing
- `video_batch_saver` - Save videos with customizable formats, codecs, and companion text files

### 🎨 LoRA Processing
- `lora_batch_loader` - Load and apply LoRAs with filtering and cycling modes

---

## Key Features
- **File-Based Filtering** - All loaders support filename pattern matching and regex
- **Training-Ready Outputs** - Maintains image-text relationships for ML datasets
- **LoRA Management** - Batch apply LoRAs with customizable strength settings
- **Flexible Modes** - Single, incremental, and random loading options
- **Memory-Efficient Frame Accumulation** - Append frames to existing files across loop iterations
- **Index Range Selection** - Select specific frame ranges for loading and saving
- **Multi-Format Support** - Supports PNG, JPG, GIF, WebP, TIFF, BMP formats

---

## Installation

Clone to your ComfyUI `custom_nodes` directory:

```
ComfyUI/custom_nodes/batch-process/
```

Restart ComfyUI

---

## Nodes Overview

### LoRA Batch Loader
Loads LoRAs from a directory with filtering options:
- **Inputs:** Model, CLIP, directory path, search filters, strength settings
- **Modes:** Single, incremental, random
- **Outputs:** Modified MODEL, CLIP, filename

### Image Batch Loader
Loads images with advanced filtering and index range selection:
- **Inputs:** Directory path, search filters, subfolder option, image list toggle, start_index, end_index
- **Outputs:** Single image, filename, image count, image list (when enabled)
- **Features:** 1-based indexing, range selection, incremental/single/random modes

### TXT Batch Loader
Loads text files with content and filename filtering:
- **Inputs:** Directory path, search filters for filename and content, start_index, end_index
- **Outputs:** Text content, filename
- **Features:** 1-based indexing, range selection

### Image Batch Saver
Saves images with companion text files and advanced options:
- **Inputs:** Images, text content, output path, naming options, append_frames, start_index, end_index
- **Features:** 
  - Frame accumulation (append_frames, defaults to false)
  - Index range selection (1-based)
  - Multi-format support (PNG, JPG, GIF, WebP, TIFF, BMP)
  - Auto-converts to GIF when append_frames is enabled with unsupported formats
  - Custom naming patterns, workflow embedding, progress tracking

### Video Batch Saver
Saves videos with customizable settings:
- **Inputs:** Videos, text content, output path, format, codec, naming options
- **Features:** Multiple formats (MP4, etc.), codec selection, preserves original FPS, companion text files, workflow metadata

### Text Modify Tool
Modifies text content programmatically:
- **Options:** Whole text, prefix, or suffix modification
- **Operations:** Search/replace, delete operations

---

## Typical Use Cases
- Preparing AI training datasets
- Batch file organization and renaming
- LoRA experimentation and comparison
- Asset management workflows
- Memory-efficient video processing with frame accumulation

---

## Requirements
- ComfyUI
- Standard ComfyUI dependencies

---

## License

This project is provided as-is for the ComfyUI community.
