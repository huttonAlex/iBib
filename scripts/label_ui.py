#!/usr/bin/env python3
"""
Minimal web UI for verifying/editing YOLO bounding boxes.

Usage:
    python scripts/label_ui.py data/unlabeled_rec0004a

Expects:
    <batch_dir>/annotated/         # images
    <batch_dir>/yolo_annotations/  # YOLO labels (class 0)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(images_dir: Path) -> list[str]:
    files = [p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def _read_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    labels: list[list[float]] = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            # class, xc, yc, w, h
            cls = int(float(parts[0]))
            if cls != 0:
                continue
            vals = [float(p) for p in parts[1:]]
            labels.append(vals)
        except ValueError:
            continue
    return labels


def _write_labels(label_path: Path, labels: list[list[float]]) -> None:
    if not labels:
        label_path.write_text("")
        return
    lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in labels]
    label_path.write_text("\n".join(lines) + "\n")


HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PointCam Label UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; display: flex; height: 100vh; }
    #sidebar { width: 320px; padding: 16px; border-right: 1px solid #ddd; overflow-y: auto; }
    #canvasWrap { flex: 1; display: flex; align-items: center; justify-content: center; background: #111; }
    #canvas { background: #111; cursor: crosshair; }
    .btn { padding: 8px 12px; margin: 4px 0; cursor: pointer; }
    .row { margin-bottom: 10px; }
    .muted { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="row"><strong>PointCam Label UI</strong></div>
    <div class="row">
      <div>Image: <span id="imgName">-</span></div>
      <div class="muted">Index: <span id="idx">0</span> / <span id="total">0</span></div>
      <div class="muted">Boxes: <span id="boxCount">0</span> <span id="dirty"></span></div>
    </div>
    <div class="row">
      <button class="btn" onclick="prevImage()">Prev (P)</button>
      <button class="btn" onclick="nextImage()">Next (N)</button>
    </div>
    <div class="row">
      <button class="btn" onclick="saveLabels()">Save (S)</button>
      <button class="btn" onclick="deleteSelected()">Delete (Del)</button>
    </div>
    <div class="row muted">
      Draw: click-drag box. Select: click box. Delete: Del. Save: S. Next/Prev: N/P.
    </div>
  </div>
  <div id="canvasWrap">
    <canvas id="canvas"></canvas>
  </div>

<script>
let images = [];
let idx = 0;
let img = new Image();
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let boxes = []; // {x1,y1,x2,y2}
let selected = -1;
let isDrawing = false;
let start = null;
let dirty = false;
let saveTimer = null;
let saving = false;

function setDirty(v) {
  dirty = v;
  document.getElementById('dirty').textContent = dirty ? '(unsaved)' : '';
}

function scheduleSave() {
  if (!dirty) return;
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveLabels();
  }, 400);
}

function fetchImages() {
  fetch('/api/list').then(r => r.json()).then(data => {
    images = data.images;
    document.getElementById('total').textContent = images.length;
    if (images.length > 0) loadImage(0);
  });
}

function loadImage(newIdx) {
  if (newIdx < 0 || newIdx >= images.length) return;
  if (dirty) saveLabels();
  idx = newIdx;
  selected = -1;
  setDirty(false);
  const name = images[idx];
  document.getElementById('idx').textContent = (idx + 1);
  document.getElementById('imgName').textContent = name;
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    fetch(`/api/labels?file=${encodeURIComponent(name)}`)
      .then(r => r.json())
      .then(data => {
        boxes = data.labels.map(l => normToPx(l));
        render();
      });
  };
  img.src = `/image/${encodeURIComponent(name)}`;
}

function normToPx(lab) {
  const [xc, yc, w, h] = lab;
  const x1 = (xc - w / 2) * canvas.width;
  const y1 = (yc - h / 2) * canvas.height;
  const x2 = (xc + w / 2) * canvas.width;
  const y2 = (yc + h / 2) * canvas.height;
  return {x1, y1, x2, y2};
}

function pxToNorm(b) {
  const x1 = Math.max(0, Math.min(b.x1, b.x2));
  const x2 = Math.min(canvas.width, Math.max(b.x1, b.x2));
  const y1 = Math.max(0, Math.min(b.y1, b.y2));
  const y2 = Math.min(canvas.height, Math.max(b.y1, b.y2));
  const w = x2 - x1;
  const h = y2 - y1;
  const xc = x1 + w / 2;
  const yc = y1 + h / 2;
  return [xc / canvas.width, yc / canvas.height, w / canvas.width, h / canvas.height];
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  boxes.forEach((b, i) => {
    ctx.strokeStyle = (i === selected) ? '#00ff88' : '#ff3355';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
  });
  document.getElementById('boxCount').textContent = boxes.length;
}

function onMouseDown(e) {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  selected = hitTest(x, y);
  if (selected >= 0) {
    render();
    return;
  }
  isDrawing = true;
  start = {x, y};
}

function onMouseMove(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  render();
  ctx.strokeStyle = '#00aaff';
  ctx.lineWidth = 2;
  ctx.strokeRect(start.x, start.y, x - start.x, y - start.y);
}

function onMouseUp(e) {
  if (!isDrawing) return;
  isDrawing = false;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const b = {x1: start.x, y1: start.y, x2: x, y2: y};
  // Ignore tiny boxes
  if (Math.abs(b.x2 - b.x1) > 4 && Math.abs(b.y2 - b.y1) > 4) {
    boxes.push(b);
    setDirty(true);
    scheduleSave();
  }
  render();
}

function hitTest(x, y) {
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const x1 = Math.min(b.x1, b.x2);
    const x2 = Math.max(b.x1, b.x2);
    const y1 = Math.min(b.y1, b.y2);
    const y2 = Math.max(b.y1, b.y2);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return i;
  }
  return -1;
}

function deleteSelected() {
  if (selected >= 0) {
    boxes.splice(selected, 1);
    selected = -1;
    setDirty(true);
    scheduleSave();
    render();
  }
}

function saveLabels() {
  if (saving) return;
  saving = true;
  const labels = boxes.map(pxToNorm);
  fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({file: images[idx], labels})
  }).then(() => {
    setDirty(false);
    saving = false;
  }).catch(() => {
    saving = false;
  });
}

function nextImage() { loadImage(Math.min(idx + 1, images.length - 1)); }
function prevImage() { loadImage(Math.max(idx - 1, 0)); }

document.addEventListener('keydown', (e) => {
  if (e.key === 'n' || e.key === 'ArrowRight') nextImage();
  if (e.key === 'p' || e.key === 'ArrowLeft') prevImage();
  if (e.key === 's') saveLabels();
  if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();
});

canvas.addEventListener('mousedown', onMouseDown);
canvas.addEventListener('mousemove', onMouseMove);
canvas.addEventListener('mouseup', onMouseUp);

fetchImages();
</script>
</body>
</html>
"""


class LabelHandler(SimpleHTTPRequestHandler):
    batch_dir: Path | None = None
    images_subdir: str = "annotated"
    images: list[str] = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
            return
        if path == "/api/list":
            self._send_json({"images": self.images})
            return
        if path == "/api/labels":
            params = urllib.parse.parse_qs(parsed.query)
            name = params.get("file", [""])[0]
            if not name:
                self._send_json({"labels": []})
                return
            label_path = self.batch_dir / "yolo_annotations" / f"{Path(name).stem}.txt"
            labels = _read_labels(label_path)
            self._send_json({"labels": labels})
            return
        if path.startswith("/image/"):
            filename = path[len("/image/") :]
            filename = urllib.parse.unquote(filename)
            img_path = self.batch_dir / self.images_subdir / filename
            if img_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(img_path.read_bytes())
                return
            self.send_error(404)
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        filename = body.get("file")
        labels = body.get("labels", [])
        if not filename:
            self._send_json({"ok": False})
            return
        label_path = self.batch_dir / "yolo_annotations" / f"{Path(filename).stem}.txt"
        _write_labels(label_path, labels)
        self._send_json({"ok": True})

    def _send_json(self, data: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick YOLO label verification UI")
    parser.add_argument("batch_dir", type=str, help="Directory with annotated/ and yolo_annotations/")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Image subdirectory name (default: annotated)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8009, help="Port to serve on")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    images_subdir = args.images_dir or "annotated"
    images_dir = batch_dir / images_subdir
    labels_dir = batch_dir / "yolo_annotations"
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"batch_dir must contain {images_subdir}/ and yolo_annotations/")

    images = _list_images(images_dir)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    LabelHandler.batch_dir = batch_dir
    LabelHandler.images_subdir = images_subdir
    LabelHandler.images = images

    server = HTTPServer((args.host, args.port), LabelHandler)
    host_for_print = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    print(f"Label UI: http://{host_for_print}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
