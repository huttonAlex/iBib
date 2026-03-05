<script>
  import { postManualCrossing } from "../api.js";
  import { pipelineState } from "../stores.js";

  let bibValue = "";
  let submitting = false;

  async function submit() {
    const bib = bibValue.trim();
    if (!bib) return;
    submitting = true;
    try {
      await postManualCrossing(bib);
      bibValue = "";
    } catch (e) {
      console.error("Manual crossing failed:", e);
    }
    submitting = false;
  }

  function onKeydown(e) {
    if (e.key === "Enter") submit();
  }

  $: disabled = $pipelineState !== "running";
</script>

<div class="bib-input" class:disabled>
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="input-icon">
    <path d="M12 4v16m8-8H4" stroke-linecap="round" stroke-linejoin="round" />
  </svg>
  <input
    type="text"
    placeholder="Manual bib #"
    bind:value={bibValue}
    on:keydown={onKeydown}
    disabled={disabled || submitting}
    inputmode="numeric"
    pattern="[0-9]*"
  />
  <button
    on:click={submit}
    disabled={disabled || submitting || !bibValue.trim()}
  >
    Add
  </button>
</div>

<style>
  .bib-input {
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    overflow: hidden;
    transition: border-color 0.15s;
  }

  .bib-input:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-dim);
  }

  .bib-input.disabled {
    opacity: 0.4;
  }

  .input-icon {
    width: 16px;
    height: 16px;
    color: var(--text-muted);
    margin-left: 0.6rem;
    flex-shrink: 0;
  }

  input {
    flex: 1;
    padding: 0.5rem 0.5rem;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-family: var(--mono);
    font-size: 0.85rem;
    min-width: 0;
  }
  input::placeholder {
    color: var(--text-muted);
    font-family: "DM Sans", system-ui, sans-serif;
  }
  input:focus {
    outline: none;
  }
  input:disabled {
    opacity: 1;
  }

  button {
    padding: 0.5rem 0.85rem;
    background: var(--accent-dim);
    border: none;
    border-left: 1px solid var(--border-default);
    color: var(--accent);
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
    transition: background 0.15s;
    white-space: nowrap;
  }
  button:hover:not(:disabled) {
    background: rgba(0, 180, 216, 0.25);
  }
  button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
