// Minimal litegraph stand-in used to exercise web/asset_filter.js without a
// browser. Run with: node tests/asset_filter_js.test.mjs
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = readFileSync(path.join(here, "..", "web", "asset_filter.js"), "utf8");

// ─── Fake litegraph graph/node ───
class FakeGraph {
    constructor() {
        this.links = new Map();
        this.nodes = new Map();
        this.nextLinkId = 1;
    }
    getNodeById(id) {
        return this.nodes.get(id) ?? null;
    }
    add(node) {
        node.graph = this;
        this.nodes.set(node.id, node);
        return node;
    }
}

class FakeNode {
    constructor(id, graph) {
        this.id = id;
        this.inputs = [];
        this.outputs = [{ name: "out", links: [] }];
        this.graph = graph;
        graph?.add(this);
    }
    addInput(name, type) {
        this.inputs.push({ name, type, link: null });
        this.onConnectionsChange?.(1, this.inputs.length - 1, false, null, null);
    }
    removeInput(index) {
        const input = this.inputs[index];
        if (input?.link != null) this.graph.links.delete(input.link);
        this.inputs.splice(index, 1);
        this.onConnectionsChange?.(1, index, false, null, null);
    }
    // origin.connect(originSlot, targetNode, targetSlot)
    connect(originSlot, target, targetSlot) {
        const input = target.inputs[targetSlot];
        if (!input) return null;
        if (input.link != null) this.graph.links.delete(input.link);
        const id = this.graph.nextLinkId++;
        this.graph.links.set(id, {
            id,
            origin_id: this.id,
            origin_slot: originSlot,
            target_id: target.id,
            target_slot: targetSlot,
        });
        input.link = id;
        target.onConnectionsChange?.(1, targetSlot, true, this.graph.links.get(id), input);
        return id;
    }
    setDirtyCanvas() {}
}

// ─── Load the extension with a stubbed `app` ───
let registered = null;
const app = {
    registerExtension(ext) {
        registered = ext;
    },
    graph: { setDirtyCanvas() {} },
};

const rafQueue = [];
globalThis.requestAnimationFrame = (fn) => {
    rafQueue.push(fn);
    return rafQueue.length;
};
function flush() {
    // Drain queued callbacks, including ones scheduled while draining.
    let guard = 0;
    while (rafQueue.length) {
        if (++guard > 1000) throw new Error("requestAnimationFrame loop did not settle");
        rafQueue.shift()();
    }
}

const moduleSource = source.replace(
    /^import\s+\{\s*app\s*\}\s+from\s+["'][^"']+["'];?\s*$/m,
    ""
);
const factory = new Function("app", `${moduleSource}\nreturn null;`);
factory(app);
assert.ok(registered, "extension should register itself");

// Apply the extension's prototype patches to our fake node class. The real
// backend declares 9 image + 3 video outputs, 0-based; the input caps are
// derived from these names.
const MAX_IMAGE = 9;
const MAX_VIDEO = 3;
const OUTPUT_NAMES = [
    ...Array.from({ length: MAX_IMAGE }, (_, i) => `image_${i}`),
    ...Array.from({ length: MAX_VIDEO }, (_, i) => `video_${i}`),
    "image_count",
    "video_count",
];
await registered.beforeRegisterNodeDef(FakeNode, {
    name: "AssetFilter",
    output_name: OUTPUT_NAMES,
});

function names(node) {
    return node.inputs.map((i) => i.name);
}

function makeFilterNode(graph) {
    const node = new FakeNode(`f${graph.nextLinkId}_${graph.nodes.size}`, graph);
    node.onAdded();
    flush();
    return node;
}

// ─── Tests ───
const tests = [];
function test(name, fn) {
    tests.push([name, fn]);
}

test("seeds one free image and one free video slot", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    assert.deepStrictEqual(names(node), ["image_0", "video_0"]);
});

test("connecting the free image slot appends exactly one new image slot", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    const src = new FakeNode("src", graph);
    src.connect(0, node, 0);
    flush();
    assert.deepStrictEqual(names(node), ["image_0", "image_1", "video_0"]);
});

test("repeated stabilize passes do not grow the input list", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    const src = new FakeNode("src", graph);
    src.connect(0, node, 0);
    flush();
    const before = names(node);
    for (let i = 0; i < 20; i++) {
        node.onConnectionsChange(1, 0, true, null, null);
        flush();
    }
    assert.deepStrictEqual(names(node), before, "layout must be stable");
});

test("indices stay compact and never drift upward", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    const a = new FakeNode("a", graph);
    const b = new FakeNode("b", graph);
    a.connect(0, node, 0);
    flush();
    b.connect(0, node, names(node).indexOf("image_1"));
    flush();
    assert.deepStrictEqual(names(node), [
        "image_0",
        "image_1",
        "image_2",
        "video_0",
    ]);
    // Disconnect the first image; the other is renumbered back to image_0.
    const link = node.inputs[0].link;
    graph.links.delete(link);
    node.inputs[0].link = null;
    node.onConnectionsChange(1, 0, false, null, null);
    flush();
    assert.deepStrictEqual(names(node), ["image_0", "image_1", "video_0"]);
});

test("image slots stay grouped above video slots", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    const v = new FakeNode("v", graph);
    v.connect(0, node, names(node).indexOf("video_0"));
    flush();
    const i = new FakeNode("i", graph);
    i.connect(0, node, names(node).indexOf("image_0"));
    flush();
    assert.deepStrictEqual(names(node), [
        "image_0",
        "image_1",
        "video_0",
        "video_1",
    ]);
    const order = names(node);
    const lastImage = order.lastIndexOf("image_1");
    const firstVideo = order.indexOf("video_0");
    assert.ok(lastImage < firstVideo, "all image slots precede video slots");
});

test("existing links survive a rebuild and point at the right origins", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    const img = new FakeNode("img", graph);
    const vid = new FakeNode("vid", graph);
    img.connect(0, node, names(node).indexOf("image_0"));
    flush();
    vid.connect(0, node, names(node).indexOf("video_0"));
    flush();
    const byName = Object.fromEntries(
        node.inputs.map((input) => [input.name, input])
    );
    const imgLink = graph.links.get(byName["image_0"].link);
    const vidLink = graph.links.get(byName["video_0"].link);
    assert.strictEqual(imgLink?.origin_id, "img");
    assert.strictEqual(vidLink?.origin_id, "vid");
});

test("multiple videos keep their own origins after renumbering", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    const v1 = new FakeNode("v1", graph);
    const v2 = new FakeNode("v2", graph);
    v1.connect(0, node, names(node).indexOf("video_0"));
    flush();
    v2.connect(0, node, names(node).indexOf("video_1"));
    flush();
    const byName = Object.fromEntries(
        node.inputs.map((input) => [input.name, input])
    );
    assert.strictEqual(graph.links.get(byName["video_0"].link)?.origin_id, "v1");
    assert.strictEqual(graph.links.get(byName["video_1"].link)?.origin_id, "v2");
    assert.deepStrictEqual(names(node), [
        "image_0",
        "video_0",
        "video_1",
        "video_2",
    ]);
});

test("loading a legacy 1-based workflow normalizes to 0-based", () => {
    const graph = new FakeGraph();
    const node = new FakeNode("loaded", graph);
    // Simulate a workflow saved before the rename: drifted 1-based names and
    // interleaving. Everything is renumbered from image_0 / video_0.
    node.addInput("image_6116", "IMAGE");
    node.addInput("video_3", "*");
    node.addInput("image_2", "IMAGE");
    const src = new FakeNode("s", graph);
    src.connect(0, node, 0);
    src.connect(0, node, 1);
    node.configure({});
    flush();
    assert.deepStrictEqual(names(node), [
        "image_0",
        "image_1",
        "video_0",
        "video_1",
    ]);
});

// Fill a zone to its cap by connecting the trailing free slot repeatedly.
function fillZone(graph, node, group, count) {
    for (let i = 0; i < count; i++) {
        const free = names(node).indexOf(`${group}_${i}`);
        if (free < 0) break;
        new FakeNode(`${group}_src${i}`, graph).connect(0, node, free);
        flush();
    }
}

test("image zone stops growing at the output count", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    fillZone(graph, node, "image", MAX_IMAGE);
    const images = names(node).filter((n) => n.startsWith("image_"));
    assert.strictEqual(images.length, MAX_IMAGE, "no slot past the last output");
    assert.strictEqual(images.at(-1), `image_${MAX_IMAGE - 1}`);
    // Every slot is connected: a full zone keeps no trailing free slot.
    const unconnected = node.inputs.filter(
        (i) => i.name.startsWith("image_") && i.link == null
    );
    assert.strictEqual(unconnected.length, 0);
});

test("video zone stops growing at its own smaller output count", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    fillZone(graph, node, "video", MAX_VIDEO);
    const videos = names(node).filter((n) => n.startsWith("video_"));
    assert.strictEqual(videos.length, MAX_VIDEO);
    assert.strictEqual(videos.at(-1), `video_${MAX_VIDEO - 1}`);
});

test("a full zone stays stable under repeated stabilize passes", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    fillZone(graph, node, "image", MAX_IMAGE);
    fillZone(graph, node, "video", MAX_VIDEO);
    const before = names(node);
    assert.strictEqual(before.length, MAX_IMAGE + MAX_VIDEO);
    for (let i = 0; i < 20; i++) {
        node.onConnectionsChange(1, 0, true, null, null);
        flush();
    }
    assert.deepStrictEqual(names(node), before);
});

test("freeing a slot in a full zone leaves the gap in place", () => {
    const graph = new FakeGraph();
    const node = makeFilterNode(graph);
    fillZone(graph, node, "image", MAX_IMAGE);
    // Disconnect image_0. The zone is already at cap, so it must not renumber:
    // outputs mirror inputs positionally, and shifting image_1 down to image_0
    // would silently move every remaining asset to a different output.
    const link = node.inputs[0].link;
    graph.links.delete(link);
    node.inputs[0].link = null;
    node.onConnectionsChange(1, 0, false, null, null);
    flush();
    const images = node.inputs.filter((i) => i.name.startsWith("image_"));
    assert.strictEqual(images.length, MAX_IMAGE, "zone stays at cap");
    const free = images.filter((i) => i.link == null);
    assert.strictEqual(free.length, 1, "exactly one free slot");
    assert.strictEqual(free[0].name, "image_0", "the freed slot itself is free");
    // Every other slot keeps the origin it had before.
    for (let i = 1; i < MAX_IMAGE; i++) {
        const input = images[i];
        assert.strictEqual(
            graph.links.get(input.link)?.origin_id,
            `image_src${i}`,
            `image_${i} kept its origin`
        );
    }
});

test("an over-cap layout from a stale workflow is trimmed to the caps", () => {
    const graph = new FakeGraph();
    const node = new FakeNode("stale", graph);
    // A workflow saved before the caps existed: 12 image inputs, all connected.
    const src = new FakeNode("s", graph);
    for (let i = 1; i <= 12; i++) node.addInput(`image_${i}`, "IMAGE");
    for (let i = 0; i < 12; i++) src.connect(0, node, i);
    node.configure({});
    flush();
    const images = names(node).filter((n) => n.startsWith("image_"));
    assert.strictEqual(images.length, MAX_IMAGE);
    assert.deepStrictEqual(
        images,
        Array.from({ length: MAX_IMAGE }, (_, i) => `image_${i}`)
    );
});

test("caps fall back to 9/3 when output names are unavailable", async () => {
    // A defensive path: if nodeData carries no output names the extension must
    // still cap rather than grow without bound.
    class Fallback extends FakeNode {}
    await registered.beforeRegisterNodeDef(Fallback, { name: "AssetFilter" });
    const graph = new FakeGraph();
    const node = new Fallback("fb", graph);
    node.onAdded();
    flush();
    fillZone(graph, node, "video", 10);
    const videos = names(node).filter((n) => n.startsWith("video_"));
    assert.strictEqual(videos.length, 3);
});

let failed = 0;
for (const [name, fn] of tests) {
    rafQueue.length = 0;
    try {
        await fn();
        console.log(`ok   ${name}`);
    } catch (error) {
        failed++;
        console.log(`FAIL ${name}\n     ${error.message}`);
    }
}
console.log(`\n${tests.length - failed}/${tests.length} passed`);
process.exit(failed ? 1 : 0);
