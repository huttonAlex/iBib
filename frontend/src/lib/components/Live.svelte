<script>
  import { onMount, onDestroy } from "svelte";
  import {
    startPipeline,
    stopPipeline,
    resetPipeline,
    getPipelineStatus,
    createWS,
  } from "../api.js";
  import {
    pipelineState,
    stats,
    crossings,
    lastError,
    addCrossing,
  } from "../stores.js";
  import StatsPanel from "./StatsPanel.svelte";
  import CameraPreview from "./CameraPreview.svelte";
  import CrossingFeed from "./CrossingFeed.svelte";
  import BibInput from "./BibInput.svelte";

  let ws;
  let statusPoll;

  onMount(async () => {
    try {
      const s = await getPipelineStatus();
      pipelineState.set(s.state);
      stats.set({
        frame_idx: s.frame_idx,
        elapsed_sec: s.elapsed_sec,
        total_crossings: s.total_crossings,
        unknown_crossings: s.unknown_crossings,
        fps: s.fps,
        total_detections: s.total_detections,
      });
      if (s.last_error) lastError.set(s.last_error);
    } catch {}

    ws = createWS((msg) => {
      if (msg.type === "crossing") {
        addCrossing(msg.data);
      } else if (msg.type === "progress") {
        stats.set(msg.data);
      }
    });

    statusPoll = setInterval(async () => {
      try {
        const s = await getPipelineStatus();
        pipelineState.set(s.state);
        if (s.last_error) lastError.set(s.last_error);
        else lastError.set(null);
      } catch {}
    }, 3000);
  });

  onDestroy(() => {
    if (ws) ws.close();
    if (statusPoll) clearInterval(statusPoll);
  });

  let actionPending = false;

  async function handleStart() {
    actionPending = true;
    try {
      await startPipeline();
      pipelineState.set("starting");
      crossings.set([]);
    } catch (e) {
      lastError.set(e.message);
    }
    actionPending = false;
  }

  async function handleStop() {
    actionPending = true;
    try {
      await stopPipeline();
      pipelineState.set("stopping");
    } catch (e) {
      lastError.set(e.message);
    }
    actionPending = false;
  }

  async function handleReset() {
    try {
      await resetPipeline();
      pipelineState.set("idle");
      lastError.set(null);
    } catch {}
  }

  $: canStart =
    ($pipelineState === "configured" || $pipelineState === "error") &&
    !actionPending;
  $: canStop =
    ($pipelineState === "running" || $pipelineState === "starting") &&
    !actionPending;
</script>

<div class="live">
  <StatsPanel />

  <div class="controls">
    <div class="control-buttons">
      {#if canStart}
        <button class="btn btn-start" on:click={handleStart}>
          <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21" /></svg>
          Start Pipeline
        </button>
      {/if}
      {#if canStop}
        <button class="btn btn-stop" on:click={handleStop}>
          <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1" /></svg>
          Stop Pipeline
        </button>
      {/if}
      {#if $pipelineState === "error"}
        <button class="btn btn-reset" on:click={handleReset}>Reset</button>
      {/if}
      {#if $pipelineState === "idle"}
        <span class="hint">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="hint-icon">
            <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Configure the pipeline in Setup, then start here
        </span>
      {/if}
    </div>

    <div class="control-right">
      <BibInput />
    </div>
  </div>

  {#if $lastError}
    <div class="error-banner">
      <div class="error-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="error-icon">
          <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <strong>Pipeline Error</strong>
      </div>
      <pre>{$lastError}</pre>
    </div>
  {/if}

  <div class="layout">
    <div class="col-video">
      <CameraPreview />
    </div>
    <div class="col-feed">
      <CrossingFeed />
    </div>
  </div>
</div>

<style>
  .live {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }

  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .control-buttons {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .control-right {
    min-width: 220px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 1rem;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn svg {
    width: 14px;
    height: 14px;
  }

  .btn-start {
    background: var(--success);
    color: #fff;
    border-color: rgba(46, 160, 67, 0.4);
  }
  .btn-start:hover {
    background: #2fb74d;
    box-shadow: 0 0 12px rgba(46, 160, 67, 0.3);
  }

  .btn-stop {
    background: var(--danger);
    color: #fff;
    border-color: rgba(218, 54, 51, 0.4);
  }
  .btn-stop:hover {
    background: #e04340;
    box-shadow: 0 0 12px rgba(218, 54, 51, 0.3);
  }

  .btn-reset {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border-color: var(--border-default);
  }
  .btn-reset:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .hint {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--text-muted);
    font-size: 0.82rem;
  }
  .hint-icon {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }

  .error-banner {
    background: var(--danger-dim);
    border: 1px solid rgba(218, 54, 51, 0.3);
    border-radius: var(--radius-md);
    padding: 0.65rem 0.85rem;
    font-size: 0.82rem;
  }
  .error-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--danger);
    margin-bottom: 0.35rem;
  }
  .error-icon {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }
  .error-banner pre {
    white-space: pre-wrap;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: #f47067;
    max-height: 150px;
    overflow-y: auto;
    line-height: 1.5;
  }

  .layout {
    display: grid;
    grid-template-columns: 1fr 360px;
    gap: 0.85rem;
    min-height: 0;
  }

  @media (max-width: 900px) {
    .layout {
      grid-template-columns: 1fr;
    }
    .control-right {
      min-width: 0;
      flex: 1;
    }
  }
</style>
