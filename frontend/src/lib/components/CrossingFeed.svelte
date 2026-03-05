<script>
  import { crossings } from "../stores.js";
</script>

<div class="feed">
  <div class="feed-header">
    <h3>Crossing Feed</h3>
    <span class="feed-count">{$crossings.length}</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th class="col-seq">#</th>
          <th class="col-bib">Bib</th>
          <th class="col-conf">Conf</th>
          <th class="col-src">Source</th>
          <th class="col-frame">Frame</th>
        </tr>
      </thead>
      <tbody>
        {#each $crossings.slice(0, 50) as c, i (c.sequence + "-" + c.frame_idx)}
          <tr class:unknown={c.bib_number === "UNKNOWN"} class:fresh={i === 0}>
            <td class="col-seq mono">{c.sequence}</td>
            <td class="col-bib mono">{c.bib_number}</td>
            <td class="col-conf mono">{c.confidence.toFixed(2)}</td>
            <td class="col-src">{c.source}</td>
            <td class="col-frame mono">{c.frame_idx}</td>
          </tr>
        {:else}
          <tr>
            <td colspan="5" class="empty">Waiting for crossings...</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>

<style>
  .feed {
    background: var(--bg-surface);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .feed-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.85rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  h3 {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
  }

  .feed-count {
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--bg-elevated);
    padding: 0.1rem 0.45rem;
    border-radius: 10px;
    min-width: 28px;
    text-align: center;
  }

  .table-wrap {
    max-height: 480px;
    overflow-y: auto;
    flex: 1;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }

  thead {
    position: sticky;
    top: 0;
    background: var(--bg-surface);
    z-index: 1;
  }

  th {
    text-align: left;
    padding: 0.4rem 0.6rem;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border-subtle);
  }

  td {
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    color: var(--text-secondary);
  }

  .mono {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
  }

  tr.fresh {
    animation: slideIn 0.25s ease-out;
  }

  tr.unknown td.col-bib {
    color: var(--danger);
    opacity: 0.7;
  }

  tr:not(.unknown) td.col-bib {
    color: var(--success-bright);
    font-weight: 600;
  }

  tr:hover td {
    background: var(--bg-elevated);
  }

  .col-seq { width: 44px; color: var(--text-muted); }
  .col-conf { width: 52px; }
  .col-src { width: 64px; }
  .col-frame { width: 64px; text-align: right; }
  th.col-frame { text-align: right; }

  .empty {
    text-align: center;
    color: var(--text-muted);
    padding: 2.5rem 1rem;
    font-size: 0.82rem;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-4px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
