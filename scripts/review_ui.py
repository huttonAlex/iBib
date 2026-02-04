#!/usr/bin/env python3
"""
Simple web-based UI for reviewing bib crop OCR predictions.

Usage:
    python scripts/review_ui.py data/unlabeled_batch1
    python scripts/review_ui.py data/unlabeled_batch2

Opens a browser where you can quickly verify/correct bib numbers.
- Shows the crop image large and clear
- Pre-fills the OCR prediction
- Press Enter to confirm, type a correction, or press 'r' to reject
- Progress is saved automatically after each action
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import threading
import webbrowser


def load_results(batch_dir: Path) -> list[dict]:
    """Load results from JSON file."""
    json_path = batch_dir / "results.json"
    with open(json_path) as f:
        return json.load(f)


def save_csv(batch_dir: Path, results: list[dict]):
    """Save updated review CSV."""
    csv_path = batch_dir / "review.csv"
    # Include expected_bibs if present in the data
    has_expected = any(r.get('expected_bibs') for r in results)
    fieldnames = [
        'source_filename', 'crop_filename', 'detection_conf',
        'ocr_prediction', 'ocr_conf', 'expected_bibs', 'verified_number', 'status'
    ] if has_expected else [
        'source_filename', 'crop_filename', 'detection_conf',
        'ocr_prediction', 'ocr_conf', 'verified_number', 'status'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def save_json(batch_dir: Path, results: list[dict]):
    """Save updated results JSON."""
    json_path = batch_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)


class ReviewHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the review UI."""

    batch_dir = None
    results = None

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.serve_app()
        elif path == '/api/stats':
            self.serve_stats()
        elif path == '/api/next':
            params = urllib.parse.parse_qs(parsed.query)
            idx = int(params.get('from', ['0'])[0])
            self.serve_next(idx)
        elif path == '/api/item':
            params = urllib.parse.parse_qs(parsed.query)
            idx = int(params.get('idx', ['0'])[0])
            self.serve_item(idx)
        elif path.startswith('/crop/'):
            self.serve_crop(path[6:])
        elif path.startswith('/source/'):
            self.serve_source(path[8:])
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/verify':
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length))
            self.handle_verify(body)
        elif parsed.path == '/api/reject':
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length))
            self.handle_reject(body)
        else:
            self.send_error(404)

    def serve_app(self):
        html = HTML_PAGE.replace('__TOTAL__', str(len(self.results)))
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_stats(self):
        verified = sum(1 for r in self.results if r.get('status') == 'verified')
        rejected = sum(1 for r in self.results if r.get('status') == 'rejected')
        pending = len(self.results) - verified - rejected
        data = {
            'total': len(self.results),
            'verified': verified,
            'rejected': rejected,
            'pending': pending,
        }
        self.send_json(data)

    def serve_next(self, from_idx):
        """Find next pending item starting from index."""
        for i in range(from_idx, len(self.results)):
            if self.results[i].get('status', 'pending') == 'pending':
                self.serve_item(i)
                return
        # Wrap around
        for i in range(0, from_idx):
            if self.results[i].get('status', 'pending') == 'pending':
                self.serve_item(i)
                return
        # All done
        self.send_json({'done': True})

    def serve_item(self, idx):
        if idx < 0 or idx >= len(self.results):
            self.send_json({'done': True})
            return

        r = self.results[idx]
        data = {
            'done': False,
            'idx': idx,
            'crop_filename': r['crop_filename'],
            'source_filename': r['source_filename'],
            'detection_conf': r.get('detection_conf', 0),
            'ocr_prediction': r.get('ocr_prediction', ''),
            'ocr_conf': r.get('ocr_conf', 0),
            'status': r.get('status', 'pending'),
            'verified_number': r.get('verified_number', ''),
            'expected_bibs': r.get('expected_bibs', ''),
        }
        self.send_json(data)

    def handle_verify(self, body):
        idx = body['idx']
        number = body['number'].strip()
        self.results[idx]['verified_number'] = number
        self.results[idx]['status'] = 'verified'

        # Auto-save every 10 verifications
        verified = sum(1 for r in self.results if r.get('status') in ('verified', 'rejected'))
        if verified % 10 == 0:
            save_csv(self.batch_dir, self.results)
            save_json(self.batch_dir, self.results)

        self.send_json({'ok': True})

    def handle_reject(self, body):
        idx = body['idx']
        self.results[idx]['status'] = 'rejected'
        self.results[idx]['verified_number'] = ''

        verified = sum(1 for r in self.results if r.get('status') in ('verified', 'rejected'))
        if verified % 10 == 0:
            save_csv(self.batch_dir, self.results)
            save_json(self.batch_dir, self.results)

        self.send_json({'ok': True})

    def serve_crop(self, filename):
        filepath = self.batch_dir / "crops" / filename
        self.serve_file(filepath)

    def serve_source(self, filename):
        # Search for source in results
        for r in self.results:
            if r['source_filename'] == filename:
                filepath = Path(r['source_image'])
                self.serve_file(filepath)
                return
        self.send_error(404)

    def serve_file(self, filepath):
        if not filepath.exists():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.end_headers()
        with open(filepath, 'rb') as f:
            self.wfile.write(f.read())

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Bib Review</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e;
    color: #eee;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: #16213e;
    border-bottom: 1px solid #333;
}
.header h1 { font-size: 18px; font-weight: 600; }
.stats {
    display: flex;
    gap: 20px;
    font-size: 14px;
}
.stats span { opacity: 0.7; }
.stats .num { opacity: 1; font-weight: 600; }
.verified-count { color: #4ade80; }
.rejected-count { color: #f87171; }
.pending-count { color: #fbbf24; }

.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    gap: 20px;
}
.image-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
}
.image-container img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
    border-radius: 8px;
    border: 2px solid #333;
}
.controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 500px;
}
.meta {
    font-size: 13px;
    opacity: 0.5;
    text-align: center;
}
.input-row {
    display: flex;
    gap: 8px;
    width: 100%;
}
#bib-input {
    flex: 1;
    font-size: 32px;
    font-weight: 700;
    text-align: center;
    padding: 12px 20px;
    border: 2px solid #4ade80;
    border-radius: 8px;
    background: #16213e;
    color: #fff;
    outline: none;
    letter-spacing: 4px;
}
#bib-input:focus { border-color: #60a5fa; }
#bib-input.rejected {
    border-color: #f87171;
    text-decoration: line-through;
    opacity: 0.5;
}
.btn-row {
    display: flex;
    gap: 8px;
    width: 100%;
}
.btn {
    flex: 1;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
}
.btn:hover { opacity: 0.85; }
.btn-confirm {
    background: #4ade80;
    color: #000;
}
.btn-reject {
    background: #f87171;
    color: #000;
}
.btn-skip {
    background: #333;
    color: #eee;
}
.shortcuts {
    font-size: 12px;
    opacity: 0.4;
    text-align: center;
}
.done-screen {
    text-align: center;
}
.done-screen h2 { font-size: 28px; margin-bottom: 12px; }
.done-screen p { opacity: 0.7; }
.progress-bar {
    width: 100%;
    height: 3px;
    background: #333;
}
.progress-fill {
    height: 100%;
    background: #4ade80;
    transition: width 0.3s;
}
</style>
</head>
<body>
<div class="header">
    <h1>Bib Review</h1>
    <div class="stats">
        <span class="verified-count"><span class="num" id="verified-count">0</span> verified</span>
        <span class="rejected-count"><span class="num" id="rejected-count">0</span> rejected</span>
        <span class="pending-count"><span class="num" id="pending-count">0</span> pending</span>
    </div>
</div>
<div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>

<div class="main" id="main">
    <div class="image-container">
        <img id="crop-img" src="" alt="Bib crop">
    </div>
    <div class="controls">
        <div class="meta" id="meta"></div>
        <div class="input-row">
            <input type="text" id="bib-input" placeholder="Bib #" autocomplete="off" maxlength="5">
        </div>
        <div class="btn-row">
            <button class="btn btn-confirm" onclick="confirm()">Confirm (Enter)</button>
            <button class="btn btn-reject" onclick="reject()">Reject (R)</button>
            <button class="btn btn-skip" onclick="skip()">Skip (S)</button>
        </div>
        <div class="shortcuts">Enter = confirm | R = reject | S = skip | Backspace = clear | Left/Right = navigate</div>
    </div>
</div>

<div class="main" id="done-screen" style="display:none">
    <div class="done-screen">
        <h2>All done!</h2>
        <p>All crops have been reviewed. Results saved automatically.</p>
        <p style="margin-top:12px">Run: <code>python scripts/process_unlabeled.py --generate-training &lt;batch_dir&gt;</code></p>
    </div>
</div>

<script>
let currentIdx = 0;
let currentItem = null;
let total = __TOTAL__;

async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    return res.json();
}

async function loadStats() {
    const s = await fetchJSON('/api/stats');
    document.getElementById('verified-count').textContent = s.verified;
    document.getElementById('rejected-count').textContent = s.rejected;
    document.getElementById('pending-count').textContent = s.pending;
    const pct = ((s.verified + s.rejected) / s.total * 100);
    document.getElementById('progress-fill').style.width = pct + '%';
}

async function loadNext() {
    const data = await fetchJSON('/api/next?from=' + currentIdx);
    if (data.done) {
        document.getElementById('main').style.display = 'none';
        document.getElementById('done-screen').style.display = 'flex';
        await loadStats();
        return;
    }
    showItem(data);
}

async function loadItem(idx) {
    const data = await fetchJSON('/api/item?idx=' + idx);
    if (data.done) {
        loadNext();
        return;
    }
    showItem(data);
}

function showItem(data) {
    currentItem = data;
    currentIdx = data.idx;
    document.getElementById('crop-img').src = '/crop/' + data.crop_filename;
    let metaText = `#${data.idx + 1}/${total} | ${data.source_filename} | det: ${data.detection_conf} | ocr conf: ${data.ocr_conf}`;
    if (data.expected_bibs) {
        metaText += ` | EXPECTED: ${data.expected_bibs}`;
    }
    document.getElementById('meta').textContent = metaText;

    const input = document.getElementById('bib-input');
    // Pre-fill with OCR prediction, or expected bib if OCR missed and only one expected
    let prefill = data.ocr_prediction || '';
    if (!prefill && data.expected_bibs && !data.expected_bibs.includes(',')) {
        prefill = data.expected_bibs;
    }
    input.value = prefill;
    input.classList.remove('rejected');
    input.select();
    input.focus();
}

async function confirm() {
    const input = document.getElementById('bib-input');
    const number = input.value.trim();
    if (!number) return;
    // Only allow digits
    if (!/^\d+$/.test(number)) return;

    await fetchJSON('/api/verify', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({idx: currentIdx, number: number})
    });
    await loadStats();
    currentIdx++;
    loadNext();
}

async function reject() {
    await fetchJSON('/api/reject', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({idx: currentIdx})
    });
    await loadStats();
    currentIdx++;
    loadNext();
}

function skip() {
    currentIdx++;
    if (currentIdx >= total) currentIdx = 0;
    loadNext();
}

document.addEventListener('keydown', (e) => {
    const input = document.getElementById('bib-input');

    // Don't intercept when actively typing digits or using backspace in input
    if (document.activeElement === input) {
        if (e.key === 'Enter') {
            e.preventDefault();
            confirm();
            return;
        }
        if (e.key === 'r' || e.key === 'R') {
            // Only reject if input is empty or we're not mid-edit
            if (input.value === '' || (input.selectionStart === 0 && input.selectionEnd === input.value.length)) {
                e.preventDefault();
                reject();
                return;
            }
            return; // Let the keypress go to the input
        }
        if (e.key === 's' || e.key === 'S') {
            if (input.value === '' || (input.selectionStart === 0 && input.selectionEnd === input.value.length)) {
                e.preventDefault();
                skip();
                return;
            }
            return;
        }
    } else {
        if (e.key === 'Enter') { e.preventDefault(); confirm(); return; }
        if (e.key === 'r' || e.key === 'R') { e.preventDefault(); reject(); return; }
        if (e.key === 's' || e.key === 'S') { e.preventDefault(); skip(); return; }
    }

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        if (currentIdx > 0) { currentIdx--; loadItem(currentIdx); }
    }
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (currentIdx < total - 1) { currentIdx++; loadItem(currentIdx); }
    }
});

// Allow only digits in input
document.getElementById('bib-input').addEventListener('input', (e) => {
    e.target.value = e.target.value.replace(/\D/g, '');
});

// Save on page close
window.addEventListener('beforeunload', () => {
    navigator.sendBeacon('/api/save', '');
});

// Start
loadNext();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Review bib OCR predictions")
    parser.add_argument("batch_dir", type=str, help="Path to batch output directory")
    parser.add_argument("--port", type=int, default=8765, help="Port for web UI")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)

    if not (batch_dir / "results.json").exists():
        print(f"Error: {batch_dir / 'results.json'} not found")
        sys.exit(1)

    # Load results
    results = load_results(batch_dir)
    print(f"Loaded {len(results)} crops")

    verified = sum(1 for r in results if r.get('status') == 'verified')
    rejected = sum(1 for r in results if r.get('status') == 'rejected')
    print(f"  Already reviewed: {verified} verified, {rejected} rejected")
    print(f"  Remaining: {len(results) - verified - rejected} pending")

    # Sort by OCR confidence descending (easiest first)
    results.sort(key=lambda x: -(x.get('ocr_conf', 0) or 0))

    # Setup handler
    ReviewHandler.batch_dir = batch_dir
    ReviewHandler.results = results

    # Start server
    server = HTTPServer(('localhost', args.port), ReviewHandler)
    url = f"http://localhost:{args.port}"
    print(f"\nReview UI running at: {url}")
    print("Press Ctrl+C to stop and save\n")

    # Open browser
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSaving...")
        save_csv(batch_dir, results)
        save_json(batch_dir, results)
        print("Saved. Goodbye!")


if __name__ == "__main__":
    main()
