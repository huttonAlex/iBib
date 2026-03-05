<script>
  import { onMount } from "svelte";
  import {
    getReviewRuns,
    getReviewQueue,
    resolveReviewItem,
    exportCrossingsUrl,
  } from "../api.js";

  let runs = [];
  let selectedRun = null;
  let queue = [];
  let currentIdx = 0;
  let correctedBib = "";
  let loading = false;

  onMount(async () => {
    try {
      runs = await getReviewRuns();
    } catch (e) {
      console.error("Failed to load runs:", e);
    }
  });

  async function loadRun(run) {
    selectedRun = run;
    loading = true;
    currentIdx = 0;
    try {
      const res = await getReviewQueue(run);
      queue = res.items || [];
    } catch (e) {
      queue = [];
    }
    loading = false;
  }

  $: current = queue[currentIdx] || null;
  $: pendingCount = queue.filter((q) => !q.review_action).length;
  $: progress = queue.length > 0 ? Math.round(((queue.length - pendingCount) / queue.length) * 100) : 0;

  function nextPending() {
    for (let i = currentIdx + 1; i < queue.length; i++) {
      if (!queue[i].review_action) { currentIdx = i; return; }
    }
    for (let i = 0; i < currentIdx; i++) {
      if (!queue[i].review_action) { currentIdx = i; return; }
    }
  }

  async function resolve(action, corrected = null) {
    if (!selectedRun || current === null) return;
    try {
      await resolveReviewItem(selectedRun, currentIdx, action, corrected);
      queue[currentIdx] = { ...queue[currentIdx], review_action: action, corrected_bib: corrected };
      queue = queue;
      nextPending();
    } catch (e) {
      console.error("Resolve failed:", e);
    }
  }

  function onKeydown(e) {
    if (!current) return;
    if (e.key === "Enter") {
      if (correctedBib.trim()) {
        resolve("correct", correctedBib.trim());
        correctedBib = "";
      } else {
        resolve("confirm");
      }
    } else if (e.key === "r" && document.activeElement.tagName !== "INPUT") {
      resolve("reject");
    } else if (e.key === "ArrowRight" || e.key === "n") {
      if (currentIdx < queue.length - 1) currentIdx++;
    } else if (e.key === "ArrowLeft" || e.key === "p") {
      if (currentIdx > 0) currentIdx--;
    }
  }
</script>

<svelte:window on:keydown={onKeydown} />

<div class="review">
  {#if !selectedRun}
    <div class="page-header">
      <h2>Review</h2>
      <p class="page-desc">Validate and correct crossing detections from past runs</p>
    </div>

    <div class="run-list">
      {#each runs as run}
        <button class="run-card" on:click={() => loadRun(run.name)} disabled={!run.has_review_queue}>
          <div class="run-info">
            <span class="run-name">{run.name}</span>
            <div class="run-badges">
              {#if run.has_crossings}
                <span class="badge">crossings</span>
              {/if}
              {#if run.has_review_queue}
                <span class="badge badge-review">review ready</span>
              {/if}
            </div>
          </div>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="run-arrow">
            <path d="M9 5l7 7-7 7" />
          </svg>
        </button>
      {:else}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="empty-icon">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <p>No runs found</p>
          <span class="empty-hint">Run the pipeline first to generate review data</span>
        </div>
      {/each}
    </div>
  {:else}
    <div class="review-header">
      <button class="btn-back" on:click={() => { selectedRun = null; queue = []; }}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M15 19l-7-7 7-7" />
        </svg>
        Back
      </button>
      <div class="header-info">
        <span class="run-title">{selectedRun}</span>
        <span class="progress-text">{queue.length - pendingCount} of {queue.length} reviewed</span>
      </div>
      <div class="header-right">
        <a href={exportCrossingsUrl(selectedRun)} class="btn-export" download>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Export CSV
        </a>
      </div>
    </div>

    <div class="progress-bar">
      <div class="progress-fill" style="width: {progress}%">
        {#if progress > 8}
          <span class="progress-label">{progress}%</span>
        {/if}
      </div>
    </div>

    {#if loading}
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading review queue...</span>
      </div>
    {:else if queue.length === 0}
      <div class="empty-state">
        <p>No items to review</p>
      </div>
    {:else if current}
      <div class="review-card" class:resolved={current.review_action}>
        <div class="card-main">
          <div class="card-nav">
            <button class="btn-nav" on:click={() => currentIdx = Math.max(0, currentIdx - 1)} disabled={currentIdx === 0}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <span class="nav-counter mono">{currentIdx + 1}<span class="nav-sep">/</span>{queue.length}</span>
            <button class="btn-nav" on:click={() => currentIdx = Math.min(queue.length - 1, currentIdx + 1)} disabled={currentIdx >= queue.length - 1}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          <div class="card-fields">
            <div class="bib-display">
              <span class="bib-label">Detected Bib</span>
              <span class="bib-number mono">{current.bib_number || current.prediction || "?"}</span>
            </div>

            <div class="meta-row">
              <div class="meta-item">
                <span class="meta-label">Confidence</span>
                <span class="meta-value mono">{(current.confidence || current.ocr_confidence || 0).toFixed(2)}</span>
              </div>
              <div class="meta-item">
                <span class="meta-label">Frame</span>
                <span class="meta-value mono">{current.frame_number || "?"}</span>
              </div>
            </div>

            {#if current.review_action}
              <div class="resolved-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="resolved-icon">
                  <path d="M5 13l4 4L19 7" />
                </svg>
                {current.review_action}{current.corrected_bib ? ` \u2192 ${current.corrected_bib}` : ""}
              </div>
            {/if}
          </div>
        </div>

        <div class="card-actions">
          <button class="btn-action btn-confirm" on:click={() => resolve("confirm")}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 13l4 4L19 7" />
            </svg>
            Confirm
            <kbd>Enter</kbd>
          </button>
          <div class="correct-row">
            <input
              type="text"
              bind:value={correctedBib}
              placeholder="Correct bib #"
              inputmode="numeric"
              class="correct-input"
            />
            <button
              class="btn-action btn-correct"
              on:click={() => { resolve("correct", correctedBib.trim()); correctedBib = ""; }}
              disabled={!correctedBib.trim()}
            >
              Correct
            </button>
          </div>
          <button class="btn-action btn-reject" on:click={() => resolve("reject")}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M6 18L18 6M6 6l12 12" />
            </svg>
            Reject
            <kbd>R</kbd>
          </button>
        </div>
      </div>

      <div class="shortcuts">
        <kbd>Enter</kbd> confirm
        <span class="shortcut-sep">&middot;</span>
        Type + <kbd>Enter</kbd> correct
        <span class="shortcut-sep">&middot;</span>
        <kbd>R</kbd> reject
        <span class="shortcut-sep">&middot;</span>
        <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate
      </div>
    {/if}
  {/if}
</div>

<style>
  .review {
    max-width: 820px;
  }

  .page-header {
    margin-bottom: 1.25rem;
  }

  h2 {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
  }

  .page-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
  }

  /* Run list */
  .run-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .run-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s, background 0.15s;
  }
  .run-card:hover:not(:disabled) {
    border-color: var(--accent);
    background: var(--bg-elevated);
  }
  .run-card:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .run-info {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .run-name {
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.88rem;
  }

  .run-badges {
    display: flex;
    gap: 0.35rem;
  }

  .badge {
    font-size: 0.65rem;
    font-weight: 500;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    background: var(--bg-elevated);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .badge-review {
    background: rgba(0, 180, 216, 0.12);
    color: var(--accent);
  }

  .run-arrow {
    width: 18px;
    height: 18px;
    color: var(--text-muted);
    flex-shrink: 0;
  }

  /* Review header */
  .review-header {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.75rem;
  }

  .btn-back {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.35rem 0.65rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 0.78rem;
    font-weight: 500;
    transition: all 0.15s;
  }
  .btn-back svg {
    width: 14px;
    height: 14px;
  }
  .btn-back:hover {
    color: var(--text-primary);
    border-color: var(--text-muted);
  }

  .header-info {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    flex: 1;
  }
  .run-title {
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.9rem;
  }
  .progress-text {
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .header-right {
    flex-shrink: 0;
  }

  .btn-export {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.75rem;
    background: var(--accent-dim);
    color: var(--accent);
    border-radius: var(--radius-sm);
    text-decoration: none;
    font-size: 0.78rem;
    font-weight: 600;
    transition: background 0.15s;
  }
  .btn-export svg {
    width: 15px;
    height: 15px;
  }
  .btn-export:hover {
    background: rgba(0, 180, 216, 0.25);
  }

  /* Progress bar */
  .progress-bar {
    height: 6px;
    background: var(--bg-elevated);
    border-radius: 3px;
    margin-bottom: 1rem;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.4s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    min-width: 0;
  }
  .progress-label {
    font-size: 0;
  }

  /* Review card */
  .review-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
    display: grid;
    grid-template-columns: 1fr 240px;
  }
  .review-card.resolved {
    border-color: rgba(46, 160, 67, 0.25);
  }

  .card-main {
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .card-nav {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .btn-nav {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-nav svg {
    width: 16px;
    height: 16px;
  }
  .btn-nav:hover:not(:disabled) {
    color: var(--text-primary);
    border-color: var(--text-muted);
  }
  .btn-nav:disabled {
    opacity: 0.25;
    cursor: not-allowed;
  }

  .nav-counter {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
  }
  .nav-sep {
    color: var(--text-muted);
    margin: 0 0.1rem;
  }

  .mono {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
  }

  .bib-display {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .bib-label {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  .bib-number {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -0.02em;
  }

  .meta-row {
    display: flex;
    gap: 1.5rem;
  }
  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .meta-label {
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  .meta-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .resolved-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--success-bright);
    background: rgba(46, 160, 67, 0.1);
    padding: 0.35rem 0.65rem;
    border-radius: var(--radius-sm);
    text-transform: capitalize;
  }
  .resolved-icon {
    width: 15px;
    height: 15px;
  }

  /* Actions panel */
  .card-actions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 1.25rem;
    background: var(--bg-elevated);
    border-left: 1px solid var(--border-subtle);
  }

  .btn-action {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 0.85rem;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-action svg {
    width: 15px;
    height: 15px;
    flex-shrink: 0;
  }
  .btn-action kbd {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 0.65rem;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.08);
    color: inherit;
    opacity: 0.6;
  }

  .btn-confirm {
    background: var(--success);
    color: #fff;
    border-color: rgba(46, 160, 67, 0.4);
  }
  .btn-confirm:hover {
    background: #2fb74d;
  }

  .btn-reject {
    background: var(--danger-dim);
    color: var(--danger);
    border-color: rgba(218, 54, 51, 0.2);
  }
  .btn-reject:hover {
    background: rgba(218, 54, 51, 0.25);
  }

  .correct-row {
    display: flex;
    gap: 0.35rem;
  }
  .correct-input {
    flex: 1;
    padding: 0.45rem 0.6rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-family: var(--mono);
    font-size: 0.85rem;
    min-width: 0;
  }
  .correct-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-dim);
  }
  .correct-input::placeholder {
    font-family: "DM Sans", system-ui, sans-serif;
    color: var(--text-muted);
    font-size: 0.78rem;
  }

  .btn-correct {
    background: var(--bg-surface);
    color: var(--warning);
    border-color: rgba(210, 153, 34, 0.25);
    white-space: nowrap;
    flex-shrink: 0;
  }
  .btn-correct:hover:not(:disabled) {
    background: rgba(210, 153, 34, 0.12);
  }
  .btn-correct:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  /* Shortcuts bar */
  .shortcuts {
    margin-top: 0.85rem;
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .shortcuts kbd {
    font-family: var(--mono);
    font-size: 0.65rem;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    color: var(--text-secondary);
  }
  .shortcut-sep {
    opacity: 0.3;
  }

  /* Empty & loading states */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 3rem 1rem;
    color: var(--text-muted);
    font-size: 0.88rem;
  }
  .empty-icon {
    width: 32px;
    height: 32px;
    opacity: 0.3;
    margin-bottom: 0.25rem;
  }
  .empty-hint {
    font-size: 0.78rem;
    opacity: 0.6;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding: 3rem;
    color: var(--text-muted);
    font-size: 0.85rem;
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

  @media (max-width: 700px) {
    .review-card {
      grid-template-columns: 1fr;
    }
    .card-actions {
      border-left: none;
      border-top: 1px solid var(--border-subtle);
    }
  }
</style>
