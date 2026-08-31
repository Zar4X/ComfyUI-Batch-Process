import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "any_batch_process.py"


def load_module():
    spec = importlib.util.spec_from_file_location("any_batch_process_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_asset(path, age_hours):
    Path(path).write_bytes(b"x")
    if age_hours:
        mtime = time.time() - age_hours * 3600.0
        os.utime(path, (mtime, mtime))


class AssetFilterTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        # Force the non-Comfy code path so tests never touch the real user
        # directory state file.
        self.module.folder_paths = None
        self.module._memory_state.clear()
        self.node = self.module.AssetFilter()
        self.tmp = tempfile.mkdtemp(prefix="asset_filter_test_")
        self.fresh_image = os.path.join(self.tmp, "fresh.png")
        self.stale_image = os.path.join(self.tmp, "stale.png")
        self.new_image = os.path.join(self.tmp, "new.png")
        self.fresh_video = os.path.join(self.tmp, "fresh.mp4")
        self.stale_video = os.path.join(self.tmp, "stale.mp4")
        make_asset(self.fresh_image, 0)
        make_asset(self.stale_image, 48)
        make_asset(self.new_image, 0)
        make_asset(self.fresh_video, 0)
        make_asset(self.stale_video, 48)
        self.image_a = torch.rand(1, 64, 64, 3)
        self.image_b = torch.rand(1, 32, 32, 3)

    def filter(self, inputs, prompt, node_id="20", workflow_id="wf-test", **payloads):
        prompt[node_id] = {
            "class_type": "AssetFilter",
            "inputs": {name: [source, 0] for name, source in inputs.items()},
        }
        return self.node.filter_assets(
            freshness_hours=24.0,
            prompt=prompt,
            node_id=node_id,
            extra_pnginfo={"workflow": {"id": workflow_id}},
            **payloads,
        )

    def split(self, result):
        """Unpack the flat output tuple into (images, videos, counts)."""
        n_img = self.module._MAX_IMAGE_OUTPUTS
        n_vid = self.module._MAX_VIDEO_OUTPUTS
        self.assertEqual(len(result), n_img + n_vid + 2)
        return (
            list(result[:n_img]),
            list(result[n_img:n_img + n_vid]),
            result[-2],
            result[-1],
        )

    def kept(self, slots):
        """Survivors in slot order, ignoring the None gaps between them."""
        return [value for value in slots if value is not None]

    def at(self, slots, index):
        """The value on output <kind>_<index> (0-based, as shown in the UI)."""
        return slots[index]

    def image_nodes(self, mapping):
        return {
            source: {"class_type": "LoadImage", "inputs": {"image": path}}
            for source, path in mapping.items()
        }

    def test_bootstrap_uses_mtime_and_positional_outputs(self):
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.stale_image})
        images, videos, image_count, video_count = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        self.assertEqual(image_count, 1)
        self.assertEqual(video_count, 0)
        kept = self.kept(images)
        self.assertEqual(len(kept), 1)
        self.assertTrue(torch.equal(kept[0], self.image_a))
        # No videos connected: every video output stays None so a reference
        # consumer skips those slots instead of being pruned.
        self.assertEqual(self.kept(videos), [])

    def test_bootstrap_all_stale_falls_back_to_pass_through(self):
        prompt = self.image_nodes({"10": self.stale_image, "11": self.stale_image})
        images, _, image_count, _ = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        self.assertEqual(image_count, 2)
        self.assertEqual(len(self.kept(images)), 2)

    def test_unchanged_name_is_filtered_on_next_run(self):
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.fresh_image})
        self.filter(
            {"image_0": "10", "image_1": "11"},
            prompt,
            image_0=self.image_a,
            image_1=self.image_b,
        )
        # Second run: slot 0 gets a new filename, slot 1 is untouched even
        # though its file mtime is fresh.
        prompt = self.image_nodes({"10": self.new_image, "11": self.fresh_image})
        images, _, image_count, _ = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        self.assertEqual(image_count, 1)
        self.assertTrue(torch.equal(self.at(images, 0), self.image_a))
        self.assertIsNone(self.at(images, 1))

    def test_rerun_without_changes_reuses_previous_selection(self):
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.stale_image})
        first = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        second = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        self.assertEqual(first[2], 1)
        self.assertEqual(second[2], 1)
        self.assertTrue(torch.equal(second[0][0], self.image_a))

    def test_runtime_assets_without_source_file_are_kept(self):
        prompt = {
            "10": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64}},
        }
        images, _, image_count, _ = self.split(
            self.filter({"image_0": "10"}, prompt, image_0=self.image_a)
        )
        self.assertEqual(image_count, 1)
        self.assertEqual(len(self.kept(images)), 1)

    def test_traces_through_intermediate_nodes(self):
        prompt = {
            "10": {"class_type": "VHS_LoadVideo", "inputs": {"video": self.fresh_video}},
            "11": {"class_type": "ImageScale", "inputs": {"image": ["10", 0]}},
            "12": {"class_type": "VHS_LoadVideo", "inputs": {"video": self.stale_video}},
        }
        video_payload = object()
        stale_payload = object()
        _, videos, _, video_count = self.split(
            self.filter(
                {"video_0": "11", "video_1": "12"},
                prompt,
                video_0=video_payload,
                video_1=stale_payload,
            )
        )
        self.assertEqual(video_count, 1)
        self.assertIs(videos[0], video_payload)

    def test_non_media_widgets_are_not_mistaken_for_sources(self):
        prompt = {
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "stale.png"}},
        }
        _, videos, _, video_count = self.split(
            self.filter({"video_0": "10"}, prompt, video_0=self.image_a)
        )
        self.assertEqual(video_count, 1)
        self.assertEqual(len(self.kept(videos)), 1)

    def test_original_sizes_are_preserved(self):
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.fresh_image})
        images, _, image_count, _ = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        self.assertEqual(image_count, 2)
        # Each survivor keeps its own resolution: separate outputs mean no
        # torch.cat, so no black-border padding.
        self.assertEqual(tuple(images[0].shape), (1, 64, 64, 3))
        self.assertEqual(tuple(images[1].shape), (1, 32, 32, 3))

    def test_flexible_inputs_accept_image_and_video_names(self):
        inputs = self.module.AssetFilter.INPUT_TYPES()["optional"]
        self.assertIn("image_0", inputs)
        self.assertIn("image_3", inputs)
        self.assertIn("video_12", inputs)
        self.assertNotIn("anything_else", inputs)
        self.assertEqual(inputs["image_0"][0], "IMAGE")
        self.assertEqual(inputs["image_3"][0], "IMAGE")
        self.assertEqual(inputs["video_12"][0], "*")

    def test_output_signature_matches_consumer_slot_counts(self):
        cls = self.module.AssetFilter
        n_img = self.module._MAX_IMAGE_OUTPUTS
        n_vid = self.module._MAX_VIDEO_OUTPUTS
        self.assertEqual(len(cls.RETURN_TYPES), n_img + n_vid + 2)
        self.assertEqual(len(cls.RETURN_NAMES), len(cls.RETURN_TYPES))
        # 0-based, matching MiniMax H3's ref_image_0..8 / ref_video_0..2.
        self.assertEqual(cls.RETURN_NAMES[0], "image_0")
        self.assertEqual(cls.RETURN_NAMES[n_img - 1], f"image_{n_img - 1}")
        self.assertEqual(cls.RETURN_NAMES[n_img], "video_0")
        self.assertEqual(cls.RETURN_NAMES[-2:], ("image_count", "video_count"))
        self.assertEqual(cls.RETURN_TYPES[:n_img], tuple(["IMAGE"] * n_img))
        self.assertEqual(cls.RETURN_TYPES[-2:], ("INT", "INT"))
        # Each output carries one asset, so the engine must not treat them as
        # lists to expand into separate executions.
        self.assertFalse(hasattr(cls, "OUTPUT_IS_LIST"))

    def test_survivors_beyond_the_last_output_are_dropped(self):
        n_img = self.module._MAX_IMAGE_OUTPUTS
        count = n_img + 2
        paths = {}
        payloads = {}
        inputs = {}
        for i in range(count):
            path = os.path.join(self.tmp, f"surplus_{i}.png")
            make_asset(path, 0)
            paths[str(i)] = path
            payloads[f"image_{i}"] = torch.rand(1, 8, 8, 3)
            inputs[f"image_{i}"] = str(i)
        images, _, image_count, _ = self.split(
            self.filter(inputs, self.image_nodes(paths), **payloads)
        )
        self.assertEqual(image_count, count)
        # The tuple keeps its declared width; slots past the last output have
        # nowhere to go, so those inputs are dropped.
        self.assertEqual(len(images), n_img)
        self.assertEqual(len(self.kept(images)), n_img)
        for i in range(n_img):
            self.assertTrue(torch.equal(self.at(images, i), payloads[f"image_{i}"]))

    def test_survivor_stays_on_its_own_output_slot(self):
        # image_0 keeps its filename (dropped), image_1 changes (kept). The
        # survivor must stay on output image_1 — compacting it onto image_0
        # would make a downstream preview show a different image than the slot
        # it is wired to.
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.stale_image})
        self.filter(
            {"image_0": "10", "image_1": "11"},
            prompt,
            image_0=self.image_a,
            image_1=self.image_b,
        )
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.new_image})
        images, _, image_count, _ = self.split(
            self.filter(
                {"image_0": "10", "image_1": "11"},
                prompt,
                image_0=self.image_a,
                image_1=self.image_b,
            )
        )
        self.assertEqual(image_count, 1)
        self.assertIsNone(self.at(images, 0))
        self.assertTrue(torch.equal(self.at(images, 1), self.image_b))

    def test_holes_are_preserved_across_the_whole_zone(self):
        # Survivors on slots 1 and 3 must appear on outputs 1 and 3, leaving
        # 0 and 2 empty. Reference consumers skip the gaps.
        paths = {}
        payloads = {}
        inputs = {}
        for i in range(4):
            path = os.path.join(self.tmp, f"hole_{i}.png")
            make_asset(path, 0 if i % 2 == 1 else 48)
            paths[str(i)] = path
            payloads[f"image_{i}"] = torch.rand(1, 8 * (i + 1), 8 * (i + 1), 3)
            inputs[f"image_{i}"] = str(i)
        images, _, image_count, _ = self.split(
            self.filter(inputs, self.image_nodes(paths), **payloads)
        )
        self.assertEqual(image_count, 2)
        self.assertIsNone(self.at(images, 0))
        self.assertIsNone(self.at(images, 2))
        self.assertTrue(torch.equal(self.at(images, 1), payloads["image_1"]))
        self.assertTrue(torch.equal(self.at(images, 3), payloads["image_3"]))

    def test_legacy_1_based_inputs_still_execute(self):
        # A workflow saved before the rename to 0-based names feeds image_1 /
        # image_2; the backend must still accept them (the frontend renumbers
        # on load, but raw execution must not break either).
        prompt = self.image_nodes({"10": self.fresh_image, "11": self.fresh_image})
        images, _, image_count, _ = self.split(
            self.filter(
                {"image_1": "10", "image_2": "11"},
                prompt,
                image_1=self.image_a,
                image_2=self.image_b,
            )
        )
        self.assertEqual(image_count, 2)
        self.assertTrue(torch.equal(self.at(images, 1), self.image_a))
        self.assertTrue(torch.equal(self.at(images, 2), self.image_b))

    def test_state_is_scoped_per_workflow_and_node(self):
        prompt = self.image_nodes({"10": self.fresh_image})
        self.filter({"image_0": "10"}, prompt, image_0=self.image_a)
        state = self.module._load_state()
        self.assertIn("wf-test:20", state)
        # A different node in a different workflow must not collide.
        prompt_b = self.image_nodes({"10": self.fresh_image})
        self.filter(
            {"image_0": "10"},
            prompt_b,
            node_id="21",
            workflow_id="wf-other",
            image_0=self.image_a,
        )
        state = self.module._load_state()
        self.assertIn("wf-test:20", state)
        self.assertIn("wf-other:21", state)

    def test_state_key_reads_nested_workflow_id(self):
        self.assertEqual(
            self.module._state_key({"workflow": {"id": "abc"}}, "7"), "abc:7"
        )
        # Tolerate a flat shape and a missing id without raising.
        self.assertEqual(self.module._state_key({"id": "flat"}, "7"), "flat:7")
        self.assertEqual(self.module._state_key(None, "7"), "None:7")

    def test_unused_outputs_are_none_not_blockers(self):
        # Only a video is connected, so every image output has nothing to emit.
        # None is required here: a reference consumer skips None slots, whereas
        # an ExecutionBlocker on any one slot prunes the whole consumer node.
        prompt = {
            "10": {"class_type": "VHS_LoadVideo", "inputs": {"video": self.fresh_video}},
        }
        images, videos, image_count, video_count = self.split(
            self.filter({"video_0": "10"}, prompt, video_0=self.image_a)
        )
        self.assertEqual(image_count, 0)
        self.assertEqual(video_count, 1)
        self.assertTrue(all(value is None for value in images))
        self.assertEqual(len(self.kept(videos)), 1)


if __name__ == "__main__":
    unittest.main()
