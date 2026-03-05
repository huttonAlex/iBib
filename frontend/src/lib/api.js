/** Fetch + WebSocket helpers for the PointCam API. */

const BASE = "";  // same origin

export async function fetchJSON(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export function getConfig() {
  return fetchJSON("/api/config");
}

export function putConfig(config) {
  return fetchJSON("/api/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

export function getModels() {
  return fetchJSON("/api/models");
}

export function uploadBibList(file) {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${BASE}/api/bib-list`, { method: "POST", body: form }).then(
    (r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }
  );
}

export function startPipeline() {
  return fetchJSON("/api/pipeline/start", { method: "POST" });
}

export function stopPipeline() {
  return fetchJSON("/api/pipeline/stop", { method: "POST" });
}

export function resetPipeline() {
  return fetchJSON("/api/pipeline/reset", { method: "POST" });
}

export function getPipelineStatus() {
  return fetchJSON("/api/pipeline/status");
}

export function postManualCrossing(bib_number) {
  return fetchJSON("/api/pipeline/manual-crossing", {
    method: "POST",
    body: JSON.stringify({ bib_number }),
  });
}

export function getReviewRuns() {
  return fetchJSON("/api/review/runs");
}

export function getReviewQueue(run) {
  return fetchJSON(`/api/review/queue?run=${encodeURIComponent(run)}`);
}

export function resolveReviewItem(run, index, action, corrected_bib = null) {
  return fetchJSON("/api/review/resolve", {
    method: "POST",
    body: JSON.stringify({ run, index, action, corrected_bib }),
  });
}

export function exportCrossingsUrl(run) {
  return `${BASE}/api/review/export?run=${encodeURIComponent(run)}`;
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

export function getNetworkStatus() {
  return fetchJSON("/api/network/status");
}

export function scanWifi() {
  return fetchJSON("/api/network/wifi/scan");
}

export function connectWifi(ssid, password) {
  return fetchJSON("/api/network/wifi/connect", {
    method: "POST",
    body: JSON.stringify({ ssid, password }),
  });
}

export function disconnectWifi(device) {
  return fetchJSON("/api/network/wifi/disconnect", {
    method: "POST",
    body: JSON.stringify({ device }),
  });
}

/** MJPEG stream URL for <img> src */
export const MJPEG_URL = `${BASE}/api/stream/mjpeg`;

/**
 * Create a reconnecting WebSocket to /api/stream/ws.
 * Returns { close, onMessage } where onMessage accepts a callback.
 */
export function createWS(onMessage) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  let ws;
  let closed = false;

  function connect() {
    ws = new WebSocket(`${proto}//${location.host}/api/stream/ws`);
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type !== "ping") onMessage(msg);
      } catch {}
    };
    ws.onclose = () => {
      if (!closed) setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();
  }

  connect();

  return {
    close() {
      closed = true;
      if (ws) ws.close();
    },
  };
}
