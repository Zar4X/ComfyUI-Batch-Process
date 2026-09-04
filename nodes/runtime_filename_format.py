"""Runtime output substitutions for ComfyUI save filename templates.

ComfyUI's built-in ``%Node.widget%`` formatter only has access to input
widgets while the prompt is being queued.  Batch loader filenames are runtime
outputs, so unresolved placeholders have to be expanded after the loader has
actually selected a file and before the saver creates its output path.
"""

from collections import OrderedDict
from functools import wraps
import re
import threading


_PLACEHOLDER_RE = re.compile(r"%([^%]+)\.([^%.]+)%")
_INVALID_FILENAME_CHARS_RE = re.compile(r'[/?<>\\:*|"\x00-\x1f\x7f]')
_MAX_PROMPTS = 128
_LOCK = threading.RLock()
_VALUES_BY_PROMPT = OrderedDict()
_LATEST_VALUES = {}


def _current_prompt_id():
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        return context.prompt_id if context is not None else None
    except (ImportError, AttributeError):
        # Older ComfyUI versions do not expose the execution context helper.
        return None


def _workflow_aliases(unique_id, extra_pnginfo):
    aliases = set()
    if unique_id is None or not isinstance(extra_pnginfo, dict):
        return aliases

    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return aliases

    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or str(node.get("id")) != str(unique_id):
            continue

        for key in ("title", "type"):
            value = node.get(key)
            if value:
                aliases.add(str(value))

        properties = node.get("properties")
        if isinstance(properties, dict):
            value = properties.get("Node name for S&R")
            if value:
                aliases.add(str(value))
        break

    return aliases


def record_runtime_output(
    node_type,
    field,
    value,
    *,
    unique_id=None,
    extra_pnginfo=None,
    display_name=None,
    prompt_id=None,
):
    """Record a node output so a saver can reference it in the same prompt."""
    install_runtime_filename_substitution()
    aliases = {str(node_type)}
    if display_name:
        aliases.add(str(display_name))
    aliases.update(_workflow_aliases(unique_id, extra_pnginfo))

    resolved_prompt_id = prompt_id if prompt_id is not None else _current_prompt_id()
    with _LOCK:
        if resolved_prompt_id is not None:
            values = _VALUES_BY_PROMPT.setdefault(str(resolved_prompt_id), {})
            _VALUES_BY_PROMPT.move_to_end(str(resolved_prompt_id))
            while len(_VALUES_BY_PROMPT) > _MAX_PROMPTS:
                _VALUES_BY_PROMPT.popitem(last=False)
        else:
            values = _LATEST_VALUES

        for alias in aliases:
            values.setdefault(alias, {})[str(field)] = value
            # ComfyUI can reuse a loader's cached output in a later prompt,
            # which means the Python function is not called again for that
            # prompt. Retaining the most recently executed value makes the
            # saver resolve to the same filename as that cached output.
            _LATEST_VALUES.setdefault(alias, {})[str(field)] = value


def replace_runtime_placeholders(text, prompt_id=None):
    """Expand recorded ``%Node.output%`` values and preserve unknown tokens."""
    if not isinstance(text, str) or "%" not in text:
        return text

    install_runtime_filename_substitution()
    resolved_prompt_id = prompt_id if prompt_id is not None else _current_prompt_id()
    with _LOCK:
        values = {}
        if resolved_prompt_id is not None:
            values.update(_VALUES_BY_PROMPT.get(str(resolved_prompt_id), {}))
        # This fallback keeps the feature usable on older, serial-execution
        # ComfyUI versions that do not expose the current prompt id.
        for alias, fields in _LATEST_VALUES.items():
            values.setdefault(alias, fields)

        def substitute(match):
            alias, field = match.groups()
            if alias not in values or field not in values[alias]:
                return match.group(0)
            value = str(values[alias][field] or "")
            return _INVALID_FILENAME_CHARS_RE.sub("_", value)

        return _PLACEHOLDER_RE.sub(substitute, text)


def install_runtime_filename_substitution():
    """Install the lightweight hook used by all ComfyUI saver nodes.

    Called lazily from record/replace instead of at import time, so loading
    this package never modifies the host application by itself.
    """
    try:
        import folder_paths
    except ImportError:
        return False

    original = folder_paths.get_save_image_path
    if getattr(original, "_batch_process_runtime_format", False):
        return True

    @wraps(original)
    def get_save_image_path(filename_prefix, *args, **kwargs):
        filename_prefix = replace_runtime_placeholders(filename_prefix)
        return original(filename_prefix, *args, **kwargs)

    get_save_image_path._batch_process_runtime_format = True
    folder_paths.get_save_image_path = get_save_image_path
    return True
