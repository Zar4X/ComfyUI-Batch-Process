# ComfyUI Batch Process
*A collection of nodes for batch processing texts, images, and LoRAs in ComfyUI*

## Core Features

### 🔧 Text Processing
- `txt_batch_loader` - Load and filter text files by filename patterns and content
- `text_modify_tool` - Edit text content with search/replace operations on prefixes, suffixes, or whole text

### 🖼️ Image Processing
- `image_batch_loader` - Load images with filtering, subfolder search, and list output options
- `image_batch_saver` - Save images with companion text files and customizable naming

### 🎨 LoRA Processing
- `lora_batch_loader` - Load and apply LoRAs with filtering and cycling modes

### 🏷️ Utilities
- `simple_image_tagger` - Generate text templates for image batches

## Key Features
- **File-Based Filtering** - All loaders support filename pattern matching and regex
- **Training-Ready Outputs** - Maintains image-text relationships for ML datasets
- **LoRA Management** - Batch apply LoRAs with customizable strength settings
- **Flexible Modes** - Single, incremental, and random loading options

## Installation

1. Clone to your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/custom_nodes/batch-process/
   ```

2. Restart ComfyUI

## Nodes Overview

### LoRA Batch Loader
Loads LoRAs from a directory with filtering options:
- **Inputs:** Model, CLIP, directory path, search filters, strength settings
- **Modes:** Single, incremental, random
- **Outputs:** Modified MODEL, CLIP, filename

### Image Batch Loader
Loads images with advanced filtering:
- **Inputs:** Directory path, search filters, subfolder option, image list toggle
- **Outputs:** Single image, filename, image count, image list (when enabled)

### TXT Batch Loader
Loads text files with content and filename filtering:
- **Inputs:** Directory path, search filters for filename and content
- **Outputs:** Text content, filename

### Image Batch Saver
Saves images with companion text files:
- **Inputs:** Images, text content, output path, naming options
- **Features:** Custom naming patterns, workflow embedding, progress tracking

### Text Modify Tool
Modifies text content programmatically:
- **Options:** Whole text, prefix, or suffix modification
- **Operations:** Search/replace, delete operations

### Simple Image Tagger
Generates text templates for image batches:
- **Inputs:** Images, text template
- **Output:** Generated text for each image

## Typical Use Cases
- Preparing AI training datasets
- Batch file organization and renaming
- LoRA experimentation and comparison
- Asset management workflows

## Requirements
- ComfyUI
- Standard ComfyUI dependencies

## License
This project is provided as-is for the ComfyUI community.

