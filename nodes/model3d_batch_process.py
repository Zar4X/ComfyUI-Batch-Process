import os
import re
import uuid
import logging

import numpy as np
import torch
from PIL import Image

import folder_paths
from server import PromptServer

try:
    from comfy_api.latest._util.geometry_types import MESH as _MESH_TYPE, File3D as _FILE3D_TYPE
except Exception:
    _MESH_TYPE = None
    _FILE3D_TYPE = None

try:
    from comfy_extras.nodes_save_3d import save_glb as _save_glb, get_mesh_batch_item as _get_mesh_batch_item
except Exception:
    _save_glb = None
    _get_mesh_batch_item = None


class _AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = _AnyType("*")

MODEL_3D_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz", ".splat", ".spz", ".ksplat"}


def _is_mesh(v):
    return _MESH_TYPE is not None and isinstance(v, _MESH_TYPE)


def _is_file3d(v):
    return _FILE3D_TYPE is not None and isinstance(v, _FILE3D_TYPE)


def _flatten_models(models):
    """Normalize any model input shape to a flat list of items.

    Each item is one of: (kind, payload) where kind ∈ {'mesh_item', 'file3d', 'path'}.
    Mesh batches are split per-row; lists are flattened.
    """
    out = []

    def push(v):
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                push(x)
            return
        if _is_mesh(v):
            if _get_mesh_batch_item is None:
                raise RuntimeError("Model3D batch IO: MESH support requires comfy_extras.nodes_save_3d")
            b = v.vertices.shape[0]
            tex = getattr(v, "texture", None)
            for i in range(b):
                verts, faces, colors, uvs = _get_mesh_batch_item(v, i)
                tex_i = tex[i] if tex is not None else None
                out.append(("mesh_item", (verts, faces, colors, uvs, tex_i)))
            return
        if _is_file3d(v):
            out.append(("file3d", v))
            return
        if isinstance(v, str):
            if os.path.splitext(v)[1].lower() in MODEL_3D_EXTS:
                out.append(("path", v))
                return
            raise ValueError(f"Model3D batch IO: string '{v}' does not look like a 3D model path")
        raise ValueError(f"Model3D batch IO: unsupported input type {type(v).__name__}")

    push(models)
    return out


def _save_one(item, dest_path, metadata=None):
    """Write a single normalized model item to dest_path. Returns the extension actually written."""
    kind, payload = item
    requested_ext = os.path.splitext(dest_path)[1].lstrip(".").lower()

    if kind == "mesh_item":
        if _save_glb is None:
            raise RuntimeError("Model3D batch IO: GLB writer unavailable")
        verts, faces, colors, uvs, tex = payload
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError("empty mesh")
        tex_img = None
        if tex is not None:
            arr = (tex.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            tex_img = Image.fromarray(arr, mode="RGB")
        # MESH path is always written as .glb regardless of the requested extension.
        if requested_ext != "glb":
            dest_path = os.path.splitext(dest_path)[0] + ".glb"
        _save_glb(verts, faces, dest_path, metadata=metadata,
                  uvs=uvs, vertex_colors=colors, texture_image=tex_img)
        return "glb", dest_path

    if kind == "file3d":
        file3d = payload
        src_ext = (file3d.format or "glb").lower()
        if requested_ext != src_ext:
            dest_path = os.path.splitext(dest_path)[0] + "." + src_ext
        file3d.save_to(dest_path)
        return src_ext, dest_path

    if kind == "path":
        src = payload
        src_ext = os.path.splitext(src)[1].lstrip(".").lower() or "glb"
        if requested_ext != src_ext:
            dest_path = os.path.splitext(dest_path)[0] + "." + src_ext
        if os.path.abspath(src) != os.path.abspath(dest_path):
            import shutil
            shutil.copy2(src, dest_path)
        return src_ext, dest_path

    raise ValueError(f"unknown item kind: {kind}")


def _normalize_list(value, count):
    if value is None:
        return [None] * count
    if not isinstance(value, list):
        return [value] * count
    if len(value) >= count:
        return value[:count]
    return value + [value[-1]] * (count - len(value))


def _resolve_output_path(base_dir, user_path):
    if not user_path or str(user_path).lower() in ("none", "."):
        return base_dir
    return user_path if os.path.isabs(str(user_path)) else os.path.join(base_dir, user_path)


def _generate_filename(prefix, suffix, padding, counter_start, counter_end, delimiter, out_dir, ext):
    try:
        existing = [f for f in os.listdir(out_dir) if f.endswith("." + ext) or f.endswith(".txt")]
    except FileNotFoundError:
        existing = []

    if suffix.strip():
        pat_start = re.compile(rf"^(\d+){re.escape(delimiter)}{re.escape(prefix)}{re.escape(delimiter)}{re.escape(suffix)}$")
        pat_end = re.compile(rf"^{re.escape(prefix)}{re.escape(delimiter)}{re.escape(suffix)}{re.escape(delimiter)}(\d+)$")
    else:
        pat_start = re.compile(rf"^(\d+){re.escape(delimiter)}{re.escape(prefix)}$")
        pat_end = re.compile(rf"^{re.escape(prefix)}{re.escape(delimiter)}(\d+)$")

    nums = []
    for f in existing:
        base = os.path.splitext(f)[0]
        match = None
        if counter_start and not counter_end:
            match = pat_start.match(base)
        elif counter_end and not counter_start:
            match = pat_end.match(base)
        elif counter_start and counter_end:
            ms, me = pat_start.match(base), pat_end.match(base)
            if ms:
                nums.append(int(ms.group(1)))
            if me:
                nums.append(int(me.group(1)))
            continue
        if match:
            nums.append(int(match.group(1)))

    counter = max(nums) + 1 if nums else 1
    parts = []
    if counter_start:
        parts.append(f"{counter:0{padding}d}")
    parts.append(str(prefix))
    if suffix.strip():
        parts.append(str(suffix))
    if counter_end:
        parts.append(f"{counter:0{padding}d}")
    return delimiter.join(parts)


def _send_progress(node_id, current, total):
    if node_id and PromptServer.instance:
        PromptServer.instance.send_sync("progress", {"node": node_id, "value": current, "max": total})


class Model3DBatchSaver:
    FUNCTION = "save"
    CATEGORY = "Batch Process"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "models": (any_type, {"tooltip": "MESH batch, File3D, or list of either / paths (e.g. AnyBatchGroup output)"}),
                "contents": ("STRING", {"forceInput": True, "tooltip": "Optional caption written next to each model as .txt"}),
                "output_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "M3D"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "filename_number_padding": ("INT", {"default": 4, "min": 1, "max": 9, "step": 1}),
                "filename_number": (["off", "start", "end"], {"default": "end"}),
                "default_extension": (["glb", "gltf", "obj", "fbx", "stl", "ply", "usdz"], {
                    "default": "glb",
                    "tooltip": "Used only for naming/collision check. File3D items keep their own extension; MESH always writes as .glb",
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(self, models=None, contents=None, output_path="", filename_prefix="M3D",
             filename_delimiter="_", filename_suffix="", filename_number_padding=4,
             filename_number="end", default_extension="glb",
             node_id=None, prompt=None, extra_pnginfo=None):
        if models is None:
            logging.warning("Model3DBatchSaver: no models provided")
            return ()

        items = _flatten_models(models)
        if not items:
            logging.warning("Model3DBatchSaver: model input flattened to zero items")
            return ()

        counter_start = filename_number == "start"
        counter_end = filename_number == "end"
        out_dir = _resolve_output_path(folder_paths.get_output_directory(), output_path)
        os.makedirs(out_dir, exist_ok=True)

        captions = _normalize_list(contents, len(items)) if contents is not None else [None] * len(items)
        original_prefix = os.path.splitext(os.path.basename(filename_prefix))[0]

        for idx, (item, caption) in enumerate(zip(items, captions)):
            try:
                base_name = _generate_filename(
                    prefix=original_prefix,
                    suffix=filename_suffix.strip("'[]"),
                    padding=filename_number_padding,
                    counter_start=counter_start,
                    counter_end=counter_end,
                    delimiter=filename_delimiter,
                    out_dir=out_dir,
                    ext=default_extension,
                )
                dest = os.path.join(out_dir, f"{base_name}.{default_extension}")
                actual_ext, actual_dest = _save_one(item, dest)
                logging.info(f"Model3DBatchSaver: wrote {actual_dest}")

                if caption is not None and str(caption).strip():
                    txt_path = os.path.splitext(actual_dest)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(str(caption).strip())
            except Exception as e:
                logging.exception(f"Model3DBatchSaver: failed item {idx}: {e}")

            _send_progress(node_id, idx + 1, len(items))

        return ()


class Model3DBatchPreview:
    FUNCTION = "preview"
    CATEGORY = "Batch Process"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "models": (any_type, {"tooltip": "MESH batch, File3D, or list of either / paths"}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    def preview(self, models=None, node_id=None):
        if models is None:
            return {"ui": {"3d": []}}

        items = _flatten_models(models)
        if not items:
            return {"ui": {"3d": []}}

        out_dir = folder_paths.get_output_directory()
        subfolder = f"preview3d_batch_{uuid.uuid4().hex[:8]}"
        full_dir = os.path.join(out_dir, subfolder)
        os.makedirs(full_dir, exist_ok=True)

        results = []
        for idx, item in enumerate(items):
            try:
                filename = f"preview_{idx:04d}.glb"
                dest = os.path.join(full_dir, filename)
                actual_ext, actual_dest = _save_one(item, dest)
                actual_filename = os.path.basename(actual_dest)
                results.append({
                    "filename": actual_filename,
                    "subfolder": subfolder,
                    "type": "output",
                })
            except Exception as e:
                logging.exception(f"Model3DBatchPreview: failed item {idx}: {e}")
            _send_progress(node_id, idx + 1, len(items))

        return {"ui": {"3d": results}}
