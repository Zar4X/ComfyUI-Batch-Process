import { app } from "../../../scripts/app.js";

// AssetFilter grows its input list on the fly: connecting the last free
// image/video slot appends a new one, and unused free slots are collapsed.
// Image inputs always stay grouped above video inputs; a newly added image
// slot appears directly under the existing image slots (same for videos).
// Input names must stay "image_N" / "video_N" so the backend can parse them.
//
// Each zone stops growing at the number of matching outputs the backend
// declares, since an input past the last output has nowhere to go. The caps are
// read from the node's own output names rather than hardcoded, so changing
// _MAX_IMAGE_OUTPUTS/_MAX_VIDEO_OUTPUTS in Python moves the frontend limit too.
//
// litegraph has no "insert input in the middle" API, so when the layout
// deviates from the canonical order the node is rebuilt: inputs are removed,
// re-added in canonical order, and links are restored through the official
// origin.connect() API.

const IMAGE_TYPE = "IMAGE";
const VIDEO_TYPE = "*";
const NAME_RE = /^(image|video)_(\d+)$/;
const GROUPS = ["image", "video"];
// Used only if the output names cannot be read (they always can in ComfyUI).
const FALLBACK_LIMITS = { image: 9, video: 3 };
const limits = { ...FALLBACK_LIMITS };

// The backend declares one image_N/video_N output per usable input slot; the
// names are 0-based, so the cap is one past the highest output index.
function readLimits(nodeData) {
    const names = nodeData?.output_name ?? nodeData?.output ?? [];
    const found = { image: 0, video: 0 };
    for (const name of names) {
        const parsed = typeof name === "string" ? parseName(name) : null;
        if (parsed) {
            found[parsed.group] = Math.max(
                found[parsed.group],
                parsed.index + 1
            );
        }
    }
    for (const group of GROUPS) {
        if (found[group] > 0) limits[group] = found[group];
    }
}

function limitFor(group) {
    return limits[group] ?? FALLBACK_LIMITS[group];
}

function typeForGroup(group) {
    return group === "image" ? IMAGE_TYPE : VIDEO_TYPE;
}

function parseName(name) {
    const match = NAME_RE.exec(name);
    return match
        ? { group: match[1], index: parseInt(match[2], 10) }
        : null;
}

function groupInputs(node, group) {
    return (node.inputs || []).filter(
        (input) => parseName(input.name)?.group === group
    );
}

// Canonical layout: image zone, then video zone. Each zone holds its
// connected inputs (renumbered 0..N-1) followed by exactly one free slot,
// capped at that zone's output count. Names are always reassigned from the
// position, so indices stay compact and can never drift upward across rebuilds.
function targetLayout(node) {
    const target = [];
    for (const group of GROUPS) {
        const limit = limitFor(group);
        const connectedCount = Math.min(
            groupInputs(node, group).filter((input) => input.link != null).length,
            limit
        );
        // One trailing free slot to grow into, unless the zone is full.
        const total = Math.min(connectedCount + 1, limit);
        for (let i = 0; i < total; i++) {
            target.push(`${group}_${i}`);
        }
    }
    return target;
}

// Names the connected inputs are expected to carry after a rebuild, in the
// same order captureConnections() returns them. Connections beyond the zone's
// cap get no name and are dropped.
function targetNamesForConnections(node) {
    const names = [];
    for (const group of GROUPS) {
        const limit = limitFor(group);
        const connected = groupInputs(node, group).filter(
            (input) => input.link != null
        );
        connected.forEach((_, i) => {
            names.push(i < limit ? `${group}_${i}` : null);
        });
    }
    return names;
}

function layoutMatches(node, target) {
    const inputs = node.inputs || [];
    return (
        inputs.length === target.length &&
        target.every((name, i) => inputs[i].name === name)
    );
}

// Connected inputs grouped image-first, each in ascending slot order. The
// order here must match targetNamesForConnections().
function captureConnections(node) {
    const connections = [];
    for (const group of GROUPS) {
        for (const input of groupInputs(node, group)) {
            if (input.link == null) continue;
            const links = node.graph?.links;
            const link = links?.get ? links.get(input.link) : links?.[input.link];
            if (!link) continue;
            const origin = node.graph?.getNodeById?.(link.origin_id);
            if (origin) {
                connections.push({ origin, slot: link.origin_slot });
            }
        }
    }
    return connections;
}

function rebuild(node, target) {
    const connections = captureConnections(node);
    const names = targetNamesForConnections(node);
    while (node.inputs.length) {
        node.removeInput(node.inputs.length - 1);
    }
    for (const name of target) {
        node.addInput(name, typeForGroup(parseName(name).group));
    }
    // Reconnect by canonical name; connections and names are index-aligned.
    // A null name means the connection sat past its zone's cap, so it is
    // deliberately left disconnected.
    connections.forEach((connection, i) => {
        if (names[i] == null) return;
        const index = node.inputs.findIndex(
            (input) => input.name === names[i]
        );
        if (index >= 0) {
            connection.origin.connect(connection.slot, node, index);
        }
    });
    node.setDirtyCanvas?.(true, true);
}

function stabilize(node) {
    if (!node.inputs || node.removed || node._assetFilterBusy) return;
    const target = targetLayout(node);
    if (layoutMatches(node, target)) return;
    // removeInput()/connect() below fire onConnectionsChange synchronously;
    // the guard keeps those callbacks from re-entering the rebuild.
    node._assetFilterBusy = true;
    try {
        rebuild(node, target);
    } finally {
        node._assetFilterBusy = false;
    }
}

const scheduled = new WeakSet();
function scheduleStabilize(node) {
    if (scheduled.has(node)) return;
    scheduled.add(node);
    requestAnimationFrame(() => {
        scheduled.delete(node);
        stabilize(node);
    });
}

app.registerExtension({
    name: "Zar4X.AssetFilter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AssetFilter") return;

        readLimits(nodeData);

        const origOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = origOnAdded?.apply(this, arguments);
            // Fresh node: seed one free image and one free video slot.
            for (const group of GROUPS) {
                if (groupInputs(this, group).length === 0) {
                    this.addInput(`${group}_0`, typeForGroup(group));
                }
            }
            return result;
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            const result = origConfigure?.apply(this, arguments);
            scheduleStabilize(this);
            return result;
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (
            type,
            slotIndex,
            isConnected,
            linkInfo,
            ioSlot
        ) {
            const result = origOnConnectionsChange?.apply(this, arguments);
            // LiteGraph NODE_INPUT === 1: only react to input connections.
            if (type === 1) scheduleStabilize(this);
            return result;
        };
    },
});
