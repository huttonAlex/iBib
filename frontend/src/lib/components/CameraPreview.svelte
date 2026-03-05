<script>
  import { MJPEG_URL } from "../api.js";
  import { pipelineState } from "../stores.js";

  let imgError = false;

  function onError() {
    imgError = true;
  }

  $: isStreaming = $pipelineState === "running";
  $: if (isStreaming) imgError = false;
</script>

<div class="preview" class:streaming={isStreaming}>
  {#if isStreaming && !imgError}
    <img src={MJPEG_URL} alt="Camera feed" on:error={onError} />
  {:else}
    <div class="placeholder">
      {#if $pipelineState === "starting"}
        <div class="loading-indicator">
          <div class="spinner"></div>
          <span>Initializing models...</span>
        </div>
      {:else if imgError}
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="placeholder-icon">
          <path d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <span>Stream unavailable</span>
        <span class="placeholder-hint">Check camera connection</span>
      {:else}
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="placeholder-icon">
          <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <span>No video feed</span>
        <span class="placeholder-hint">Start the pipeline to begin streaming</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .preview {
    background: #000;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    aspect-ratio: 16 / 9;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .preview.streaming {
    border-color: rgba(63, 185, 80, 0.25);
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 2rem;
    text-align: center;
  }

  .placeholder-icon {
    width: 36px;
    height: 36px;
    opacity: 0.3;
    margin-bottom: 0.25rem;
  }

  .placeholder-hint {
    font-size: 0.75rem;
    color: var(--text-muted);
    opacity: 0.6;
  }

  .loading-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
  }

  .spinner {
    width: 28px;
    height: 28px;
    border: 2px solid var(--border-default);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
