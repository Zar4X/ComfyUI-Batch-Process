# ComfyUI Batch Process   
*A collection of nodes for batch processing texts and images in ComfyUI*

---

## Core Features

### 🔧 Text Processing Tools
- `text_modify_tool` - Edit filenames, content, and prefixes/suffixes (replace/delete operations)
- `txt_batch_loader` - Batch load + filter text files using filename patterns
- `simple_image_tagger` - Bulk create/update text labels for images

### 🖼️ Image Processing Tools
- `image_batch_loader` - Load and filter images by filename criteria
- `image_batch_saver` - Batch save images with companion text files
- `image_list_loader` - Load pre-defined image sequences

---

## Key Features
- **File-Based Filtering**   
  `▸` All loaders support filename pattern matching   
  `▸` Regular expression compatible
    
- **Training-Ready Outputs**   
  `▸` Maintains image-text relationships for ML datasets   
  `▸` Supports independent text/image batch saving
    
---

## Typical Use Cases
Preparing AI training datasets:
○ Bulk organize generated images
○ Synchronize text annotations
○ Create structured datasets

Asset management:
○ Batch rename project files
○ Update metadata across file groups
○ Filter content by naming patterns


---

## Workflow Integration
Works natively with ComfyUI's node-based architecture:
[Input] ➔ [Batch Loader] ➔ [Processing Nodes] ➔ [Batch Saver] ➔ [Output]

