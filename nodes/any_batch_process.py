import json
import os
import re
import time

import torch
import torch.nn.functional as F

try:
    import comfy.utils
except Exception:
    comfy = None

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from comfy_api.latest._util.geometry_types import MESH as _MESH_TYPE, File3D as _FILE3D_TYPE
except Exception:
    _MESH_TYPE = None
    _FILE3D_TYPE = None

try:
    from comfy_extras.nodes_save_3d import pack_variable_mesh_batch as _pack_mesh_batch, get_mesh_batch_item as _get_mesh_batch_item
except Exception:
    _pack_mesh_batch = None
    _get_mesh_batch_item = None


class _AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = _AnyType("*")


def _to_bhwc(tensor):
    if tensor.dim() == 3:
        # HWC -> BHWC
        return tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError(
            f"Only 3D/4D image tensors are supported, got shape {tuple(tensor.shape)}"
        )
    # If looks like BCHW, convert to BHWC
    if tensor.shape[1] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
        return tensor.movedim(1, -1)
    return tensor


def _pad_to_size_center(tensor, target_h, target_w):
    # tensor: BHWC
    b, h, w, c = tensor.shape
    if h == target_h and w == target_w:
        return tensor

    out = torch.zeros(
        (b, target_h, target_w, c), dtype=tensor.dtype, device=tensor.device
    )
    top = (target_h - h) // 2
    left = (target_w - w) // 2
    out[:, top : top + h, left : left + w, :] = tensor
    return out


def _batch_tensors_with_padding(tensors):
    converted = [_to_bhwc(t) for t in tensors]
    channels = converted[0].shape[-1]
    if any(t.shape[-1] != channels for t in converted):
        raise ValueError("Channel mismatch in tensor list")

    max_h = max(t.shape[1] for t in converted)
    max_w = max(t.shape[2] for t in converted)
    padded = [_pad_to_size_center(t, max_h, max_w) for t in converted]
    return torch.cat(padded, dim=0)


class AnyBatchGroup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_1": (any_type, {}),
                "any_2": (any_type, {}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("batch",)
    FUNCTION = "batch"
    CATEGORY = "Batch Process"

    def _common_upscale(self, tensor, width, height):
        if comfy is not None:
            return comfy.utils.common_upscale(
                tensor, width, height, "bilinear", "center"
            )
        # Fallback for non-Comfy test environments.
        return F.interpolate(tensor, size=(height, width), mode="bilinear")

    def latent_batch(self, any_1, any_2):
        samples_out = any_1.copy()
        s1 = any_1["samples"]
        s2 = any_2["samples"]

        if s1.shape[1:] != s2.shape[1:]:
            s2 = self._common_upscale(s2, s1.shape[3], s1.shape[2])
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = any_1.get(
            "batch_index", [x for x in range(0, s1.shape[0])]
        ) + any_2.get("batch_index", [x for x in range(0, s2.shape[0])])

        return samples_out

    def _to_bhwc(self, tensor):
        return _to_bhwc(tensor)

    def _pad_to_size_center(self, tensor, target_h, target_w):
        return _pad_to_size_center(tensor, target_h, target_w)

    def _is_list_of_tensors(self, value):
        return isinstance(value, list) and len(value) > 0 and all(
            isinstance(x, torch.Tensor) for x in value
        )

    def _batch_tensors_with_padding(self, tensors):
        return _batch_tensors_with_padding(tensors)

    def _is_mesh(self, value):
        return _MESH_TYPE is not None and isinstance(value, _MESH_TYPE)

    def _is_file3d(self, value):
        return _FILE3D_TYPE is not None and isinstance(value, _FILE3D_TYPE)

    def _contains_file3d(self, value):
        if self._is_file3d(value):
            return True
        if isinstance(value, list):
            return any(self._is_file3d(x) for x in value)
        return False

    def _contains_mesh(self, value):
        if self._is_mesh(value):
            return True
        if isinstance(value, list):
            return any(self._is_mesh(x) for x in value)
        return False

    def _to_file3d_list(self, value):
        if self._is_file3d(value):
            return [value]
        if isinstance(value, list):
            return [x for x in value if self._is_file3d(x)]
        return []

    def _mesh_to_list(self, mesh):
        # Unpack a (possibly variable-length) MESH into a list of single-item MESH objects.
        out = []
        b = mesh.vertices.shape[0]
        tex = getattr(mesh, "texture", None)
        for i in range(b):
            v, f, c, u = _get_mesh_batch_item(mesh, i)
            t_i = tex[i:i+1] if tex is not None else None
            out.append(_MESH_TYPE(v.unsqueeze(0), f.unsqueeze(0),
                                  uvs=u.unsqueeze(0) if u is not None else None,
                                  vertex_colors=c.unsqueeze(0) if c is not None else None,
                                  texture=t_i))
        return out

    def _concat_meshes(self, a, b):
        if _pack_mesh_batch is None or _get_mesh_batch_item is None:
            raise RuntimeError("AnyBatchGroup: MESH batching requires comfy_extras.nodes_save_3d helpers")
        vertices, faces, colors, uvs, textures = [], [], [], [], []
        has_colors = has_uvs = has_tex = False
        for m in (a, b):
            mb = m.vertices.shape[0]
            tex = getattr(m, "texture", None)
            for i in range(mb):
                v, f, c, u = _get_mesh_batch_item(m, i)
                vertices.append(v)
                faces.append(f)
                colors.append(c)
                uvs.append(u)
                if tex is not None:
                    textures.append(tex[i])
                    has_tex = True
                if c is not None:
                    has_colors = True
                if u is not None:
                    has_uvs = True

        if has_colors and any(c is None for c in colors):
            raise ValueError("AnyBatchGroup: cannot batch meshes — some have vertex_colors and some don't")
        if has_uvs and any(u is None for u in uvs):
            raise ValueError("AnyBatchGroup: cannot batch meshes — some have uvs and some don't")
        if has_tex and len(textures) != len(vertices):
            raise ValueError("AnyBatchGroup: cannot batch meshes — texture coverage is partial")

        packed_tex = torch.stack(textures, dim=0) if has_tex else None
        return _pack_mesh_batch(
            vertices, faces,
            colors=colors if has_colors else None,
            uvs=uvs if has_uvs else None,
            texture=packed_tex,
        )

    def batch(self, any_1, any_2):
        # 3D model batching: MESH ↔ MESH, File3D ↔ File3D / list[File3D]. Mixed raises.
        a_is_mesh, b_is_mesh = self._contains_mesh(any_1), self._contains_mesh(any_2)
        a_is_f3d, b_is_f3d = self._contains_file3d(any_1), self._contains_file3d(any_2)
        if (a_is_mesh and b_is_f3d) or (a_is_f3d and b_is_mesh):
            raise ValueError(
                "AnyBatchGroup: cannot mix MESH and File3D in one batch — convert to a single type first"
            )
        if a_is_mesh or b_is_mesh:
            if any_1 is None:
                return (any_2,)
            if any_2 is None:
                return (any_1,)
            meshes_1 = [any_1] if self._is_mesh(any_1) else [m for m in any_1 if self._is_mesh(m)]
            meshes_2 = [any_2] if self._is_mesh(any_2) else [m for m in any_2 if self._is_mesh(m)]
            merged = meshes_1[0]
            for m in meshes_1[1:] + meshes_2:
                merged = self._concat_meshes(merged, m)
            return (merged,)
        if a_is_f3d or b_is_f3d:
            if any_1 is None:
                return (self._to_file3d_list(any_2) or any_2,)
            if any_2 is None:
                return (self._to_file3d_list(any_1) or any_1,)
            return (self._to_file3d_list(any_1) + self._to_file3d_list(any_2),)

        # Some any-type chains may pass list-of-tensors; normalize and batch consistently.
        if self._is_list_of_tensors(any_1) or self._is_list_of_tensors(any_2):
            list_1 = any_1 if self._is_list_of_tensors(any_1) else [any_1]
            list_2 = any_2 if self._is_list_of_tensors(any_2) else [any_2]
            tensors = [t for t in (list_1 + list_2) if isinstance(t, torch.Tensor)]
            if len(tensors) == 0:
                return (any_1 if any_2 is None else any_2,)
            return (self._batch_tensors_with_padding(tensors),)

        if isinstance(any_1, torch.Tensor) or isinstance(any_2, torch.Tensor):
            if any_1 is None:
                return (any_2,)
            if any_2 is None:
                return (any_1,)

            t1 = self._to_bhwc(any_1)
            t2 = self._to_bhwc(any_2)

            if t1.shape[-1] != t2.shape[-1]:
                raise ValueError(
                    f"Channel mismatch: {t1.shape[-1]} vs {t2.shape[-1]}"
                )

            # Same size: normal concat.
            if t1.shape[1:] == t2.shape[1:]:
                return (torch.cat((t1, t2), 0),)

            # Different size: pad to max resolution with black borders, then concat.
            max_h = max(t1.shape[1], t2.shape[1])
            max_w = max(t1.shape[2], t2.shape[2])
            t1 = self._pad_to_size_center(t1, max_h, max_w)
            t2 = self._pad_to_size_center(t2, max_h, max_w)
            return (torch.cat((t1, t2), 0),)
        elif isinstance(any_1, (str, float, int)):
            if any_2 is None:
                return (any_1,)
            if isinstance(any_2, tuple):
                return (any_2 + (any_1,),)
            if isinstance(any_2, list):
                return (any_2 + [any_1],)
            return ([any_1, any_2],)
        elif isinstance(any_2, (str, float, int)):
            if any_1 is None:
                return (any_2,)
            if isinstance(any_1, tuple):
                return (any_1 + (any_2,),)
            if isinstance(any_1, list):
                return (any_1 + [any_2],)
            return ([any_2, any_1],)
        elif isinstance(any_1, dict) and "samples" in any_1:
            if any_2 is None:
                return (any_1,)
            if isinstance(any_2, dict) and "samples" in any_2:
                return (self.latent_batch(any_1, any_2),)
        elif isinstance(any_2, dict) and "samples" in any_2:
            if any_1 is None:
                return (any_2,)
            if isinstance(any_1, dict) and "samples" in any_1:
                return (self.latent_batch(any_2, any_1),)
        else:
            if any_1 is None:
                return (any_2,)
            if any_2 is None:
                return (any_1,)
            return (any_1 + any_2,)


_ASSET_INPUT_RE = re.compile(r"^(image|video)_(\d+)$")

# Extensions accepted when tracing an input back to a source file. Keeps
# arbitrary STRING widgets (captions etc.) from being mistaken for assets.
_MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif",
    ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".mpg", ".mpeg",
}


class _AssetFilterInputs(dict):
    """Flexible optional inputs for AssetFilter.

    ComfyUI asks this dict whether a queued input name is valid and which
    type it expects. ``image_N`` slots accept IMAGE, ``video_N`` slots accept
    anything (video payloads may be IMAGE frame batches or VIDEO objects,
    depending on the loader). The frontend grows/shrinks the input list, so
    no fixed input names are declared.
    """

    def __contains__(self, key):
        return _ASSET_INPUT_RE.match(key) is not None

    def __getitem__(self, key):
        match = _ASSET_INPUT_RE.match(key)
        if match is None:
            raise KeyError(key)
        return ("IMAGE",) if match.group(1) == "image" else ("*",)

    def __iter__(self):
        return iter(())

    def keys(self):
        return iter(())


def _has_media_extension(path):
    return os.path.splitext(path)[1].lower() in _MEDIA_EXTENSIONS


def _resolve_source_file(value):
    """Return the absolute path if a widget string points at an existing media file."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()

    candidates = []
    if folder_paths is not None:
        # Handles ComfyUI's annotated widget values, e.g. "clip.mp4 [input]".
        try:
            candidates.append(folder_paths.get_annotated_filepath(value))
        except Exception:
            pass
    candidates.append(value)

    roots = []
    if folder_paths is not None:
        for getter in (
            folder_paths.get_input_directory,
            folder_paths.get_output_directory,
            folder_paths.get_temp_directory,
        ):
            try:
                roots.append(getter())
            except Exception:
                pass

    for candidate in candidates:
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and _has_media_extension(candidate):
                return candidate
            continue
        for root in roots:
            full = os.path.join(root, candidate)
            if os.path.isfile(full) and _has_media_extension(full):
                return full
    return None


def _trace_source_file(prompt, node_id, input_name, max_depth=8):
    """Walk upstream from an input until some node exposes a widget value that
    resolves to an existing media file. Returns ``(resolved_path, widget_value)``;
    both are None when the input is not file-backed (produced by the run)."""
    if not isinstance(prompt, dict) or node_id is None or input_name is None:
        return None, None

    node_id = str(node_id)
    visited = set()
    queue = [(node_id, input_name, 0)]
    while queue:
        current_id, current_input, depth = queue.pop(0)
        if depth > max_depth:
            continue
        node = prompt.get(current_id)
        if not isinstance(node, dict):
            continue
        link = node.get("inputs", {}).get(current_input)
        if not isinstance(link, (list, tuple)) or len(link) < 1:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = prompt.get(upstream_id)
        if not isinstance(upstream, dict):
            continue
        inputs = upstream.get("inputs", {})
        # Check this node's own widget values before going deeper.
        for value in inputs.values():
            if isinstance(value, str):
                resolved = _resolve_source_file(value)
                if resolved is not None:
                    return resolved, value
        for key, value in inputs.items():
            if isinstance(value, (list, tuple)):
                queue.append((upstream_id, key, depth + 1))
    return None, None


_STATE_FILENAME = "zar4x_asset_filter_state.json"
_STATE_MAX_ENTRIES = 64
_RUNTIME_ASSET = "__runtime__"
_memory_state = {}


def _state_path():
    if folder_paths is None:
        return None
    for getter in (folder_paths.get_user_directory, folder_paths.get_temp_directory):
        try:
            return os.path.join(getter(), _STATE_FILENAME)
        except Exception:
            continue
    return None


def _load_state():
    path = _state_path()
    if path is None:
        return _memory_state
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _save_state(state):
    if not isinstance(state, dict):
        return
    if len(state) > _STATE_MAX_ENTRIES:
        # Drop oldest entries; dict order is insertion order.
        for key in list(state)[: len(state) - _STATE_MAX_ENTRIES]:
            state.pop(key, None)
    path = _state_path()
    if path is None:
        # state may be _memory_state itself; only copy when it is not.
        if state is not _memory_state:
            _memory_state.clear()
            _memory_state.update(state)
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except Exception as error:
        print(f"[AssetFilter] warning: could not persist state to {path}: {error}")


def _state_key(extra_pnginfo, node_id):
    # ComfyUI sends extra_pnginfo as {"workflow": {...}}; the stable workflow
    # uuid lives inside that nested dict.
    workflow_id = None
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if isinstance(workflow, dict):
            workflow_id = workflow.get("id")
        if workflow_id is None:
            workflow_id = extra_pnginfo.get("id")
    return f"{workflow_id}:{node_id}"


_MAX_IMAGE_OUTPUTS = 9
_MAX_VIDEO_OUTPUTS = 3


class AssetFilter:
    """Drops stale asset slots left over from previous runs.

    Connect reference images/videos (e.g. LoadImage, VHS_LoadVideo) to the
    growing input list. A slot is considered new when its source filename
    changed since this node last ran; unchanged slots are filtered out.
    On the very first run (no saved state) the file modification time within
    `freshness_hours` decides instead, and re-running without any change
    reuses the previous run's selection.

    Outputs mirror the inputs positionally: input image_N always leaves on
    output image_N, so a slot's identity never shifts and previews cannot swap
    images between runs. Outputs whose input was filtered (or is unconnected)
    return None. Reference consumers such as MiniMax H3 skip None slots, so
    wiring image_0..8 into ref_image_0..8 once and leaving it wired works
    regardless of how many assets survive — the consumer still runs exactly
    once with only the survivors, and gaps in the middle are harmless.

    Note that None will crash consumers that do not tolerate it (PreviewImage,
    SaveImage). Feed those from a specific slot you know is populated, or gate
    them on image_count.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "freshness_hours": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.1, "max": 720.0, "step": 0.5},
                ),
            },
            "optional": _AssetFilterInputs(),
            "hidden": {
                "prompt": "PROMPT",
                "node_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (
        tuple(["IMAGE"] * _MAX_IMAGE_OUTPUTS)
        + tuple([any_type] * _MAX_VIDEO_OUTPUTS)
        + ("INT", "INT")
    )
    RETURN_NAMES = (
        tuple(f"image_{i}" for i in range(_MAX_IMAGE_OUTPUTS))
        + tuple(f"video_{i}" for i in range(_MAX_VIDEO_OUTPUTS))
        + ("image_count", "video_count")
    )
    FUNCTION = "filter_assets"
    CATEGORY = "Batch Process"

    def filter_assets(self, freshness_hours=24.0, prompt=None, node_id=None,
                      extra_pnginfo=None, **kwargs):
        assets = []
        for name, value in kwargs.items():
            match = _ASSET_INPUT_RE.match(name)
            if match is None:
                continue
            assets.append((match.group(1), int(match.group(2)), value))
        assets.sort(key=lambda item: (item[0], item[1]))

        window = float(freshness_hours) * 3600.0
        now = time.time()

        state = _load_state()
        key = _state_key(extra_pnginfo, node_id)
        prev = state.get(key)
        if not isinstance(prev, dict):
            prev = None

        entries = []
        for kind, index, value in assets:
            slot = f"{kind}_{index}"
            resolved, widget = _trace_source_file(prompt, node_id, slot)
            # The widget value is the filename the user sees; use it as the
            # change-detection identity.
            name = widget if isinstance(widget, str) and widget.strip() else (
                resolved or _RUNTIME_ASSET
            )
            entries.append({
                "slot": slot,
                "kind": kind,
                "index": index,
                "value": value,
                "resolved": resolved,
                "name": name,
                "kept": None,
                "reason": None,
            })

        all_unchanged = prev is not None and set(prev.keys()) == {
            e["slot"] for e in entries
        } and all(prev[e["slot"]].get("name") == e["name"] for e in entries)

        for entry in entries:
            slot, resolved, name = entry["slot"], entry["resolved"], entry["name"]
            runtime = resolved is None
            if prev is None:
                # First run with no saved state: fall back to mtime.
                kept = runtime
                if not kept:
                    try:
                        kept = (now - os.path.getmtime(resolved)) <= window
                    except OSError:
                        kept = True
                reason = "bootstrap-fresh" if kept else "bootstrap-stale"
            elif all_unchanged:
                # Pure re-run: reuse the previous selection.
                kept = bool(prev[slot].get("kept"))
                reason = "reused"
            else:
                prev_name = prev.get(slot, {}).get("name")
                kept = runtime or name != prev_name
                reason = "changed" if kept else "unchanged"
            entry["kept"] = kept
            entry["reason"] = reason

        if entries and not any(e["kept"] for e in entries):
            print(
                "[AssetFilter] warning: every connected asset looks stale — "
                "passing all of them through as a fallback."
            )
            for entry in entries:
                entry["kept"] = True

        images = {
            e["index"]: e["value"]
            for e in entries
            if e["kind"] == "image" and e["kept"]
        }
        videos = {
            e["index"]: e["value"]
            for e in entries
            if e["kind"] == "video" and e["kept"]
        }

        for entry in entries:
            print(
                f"[AssetFilter] {entry['slot']}: {entry['reason']} "
                f"({'keep' if entry['kept'] else 'drop'}) "
                f"<- {entry['resolved'] or entry['name']}"
            )

        new_entry = {
            e["slot"]: {"name": e["name"], "kept": e["kept"]} for e in entries
        }
        if prev is None or not all_unchanged:
            state[key] = new_entry
            _save_state(state)

        total_images = sum(1 for e in entries if e["kind"] == "image")
        total_videos = sum(1 for e in entries if e["kind"] == "video")
        print(
            f"[AssetFilter] kept {len(images)}/{total_images} images, "
            f"{len(videos)}/{total_videos} videos"
        )
        self._warn_out_of_range(images, _MAX_IMAGE_OUTPUTS, "image")
        self._warn_out_of_range(videos, _MAX_VIDEO_OUTPUTS, "video")
        # Mirror inputs positionally: input image_N leaves on output image_N so
        # a slot's identity never shifts between runs. Filtered and unconnected
        # slots emit None; reference consumers skip None, and a gap in the
        # middle is harmless to them. An ExecutionBlocker would instead prune
        # the whole downstream node, not just that one slot.
        return (
            self._place(images, _MAX_IMAGE_OUTPUTS)
            + self._place(videos, _MAX_VIDEO_OUTPUTS)
            + (len(images), len(videos))
        )

    @staticmethod
    def _place(by_index, count):
        """Put each survivor on the output matching its 0-based input index."""
        return tuple(by_index.get(i) for i in range(count))

    @staticmethod
    def _warn_out_of_range(by_index, count, kind):
        extra = sorted(i for i in by_index if i >= count)
        if extra:
            slots = ", ".join(f"{kind}_{i}" for i in extra)
            print(
                f"[AssetFilter] warning: {slots} survived but only "
                f"{count} {kind} outputs exist — dropping them."
            )
