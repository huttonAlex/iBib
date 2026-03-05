<script>
  import { stats, pipelineState, crossings } from "../stores.js";

  function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  $: known = $stats.total_crossings - $stats.unknown_crossings;
  $: uniqueBibs = new Set(
    $crossings
      .filter((c) => c.bib_number !== "UNKNOWN")
      .map((c) => c.bib_number)
  ).size;
</script>

<div class="stats-bar">
  <div class="stat-group">
    <div class="stat">
      <span class="stat-label">Status</span>
      <span class="stat-value">
        <span class="status-dot status-dot-{$pipelineState}"></span>
        <span class="state-text state-{$pipelineState}">{$pipelineState}</span>
      </span>
    </div>
    <div class="stat-sep"></div>
    <div class="stat">
      <span class="stat-label">Frames</span>
      <span class="stat-value mono">{$stats.frame_idx.toLocaleString()}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Elapsed</span>
      <span class="stat-value mono">{fmtTime($stats.elapsed_sec)}</span>
    </div>
    <div class="stat">
      <span class="stat-label">FPS</span>
      <span class="stat-value mono">{$stats.fps.toFixed(1)}</span>
    </div>
  </div>

  <div class="stat-group crossing-stats">
    <div class="stat">
      <span class="stat-label">Crossings</span>
      <span class="stat-value mono">{$stats.total_crossings}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Known</span>
      <span class="stat-value mono val-success">{known}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Unknown</span>
      <span class="stat-value mono val-warning">{$stats.unknown_crossings}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Unique</span>
      <span class="stat-value mono val-accent">{uniqueBibs}</span>
    </div>
  </div>
</div>

<style>
  .stats-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    flex-wrap: wrap;
  }

  .stat-group {
    display: flex;
    align-items: center;
    gap: 1.25rem;
  }

  .stat-sep {
    width: 1px;
    height: 24px;
    background: var(--border-default);
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.15rem;
    min-width: 52px;
  }

  .stat-label {
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
  }

  .stat-value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.35rem;
    line-height: 1;
  }

  .mono {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
  }

  .val-success { color: var(--success-bright); }
  .val-warning { color: var(--warning); }
  .val-accent { color: var(--accent); }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .status-dot-running {
    background: var(--success-bright);
    box-shadow: 0 0 6px rgba(63, 185, 80, 0.6);
    animation: pulse 2s ease-in-out infinite;
  }
  .status-dot-starting,
  .status-dot-stopping {
    background: var(--warning);
    box-shadow: 0 0 6px rgba(210, 153, 34, 0.6);
    animation: pulse 1s ease-in-out infinite;
  }
  .status-dot-error {
    background: var(--danger);
    box-shadow: 0 0 6px rgba(218, 54, 51, 0.6);
  }
  .status-dot-idle {
    background: var(--text-muted);
  }
  .status-dot-configured {
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent-glow);
  }

  .state-text {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: capitalize;
  }
  .state-running { color: var(--success-bright); }
  .state-starting, .state-stopping { color: var(--warning); }
  .state-error { color: var(--danger); }
  .state-idle { color: var(--text-muted); }
  .state-configured { color: var(--accent); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  @media (max-width: 700px) {
    .stats-bar {
      gap: 0.5rem;
    }
    .stat-group {
      gap: 0.75rem;
    }
    .stat-sep { display: none; }
  }
</style>
