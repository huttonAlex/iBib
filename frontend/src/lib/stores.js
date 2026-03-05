import { writable } from "svelte/store";

/** Current pipeline state string: idle, configured, starting, running, stopping, error */
export const pipelineState = writable("idle");

/** Array of recent crossing events (newest first, max 200) */
export const crossings = writable([]);

/** Live stats from WebSocket progress events */
export const stats = writable({
  frame_idx: 0,
  elapsed_sec: 0,
  total_crossings: 0,
  unknown_crossings: 0,
  fps: 0,
  total_detections: 0,
});

/** Pipeline configuration object */
export const config = writable({});

/** Last error message */
export const lastError = writable(null);

/** Helper to add a crossing and cap the array */
export function addCrossing(event) {
  crossings.update((arr) => {
    const next = [event, ...arr];
    if (next.length > 200) next.length = 200;
    return next;
  });
}
