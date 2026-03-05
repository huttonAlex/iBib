<script>
  import { onMount } from "svelte";
  import {
    getConfig,
    putConfig,
    getModels,
    uploadBibList,
  } from "../api.js";
  import { config, pipelineState } from "../stores.js";

  let models = { detectors: [], ocr: [], pose: [] };
  let bibFile = null;
  let bibCount = null;
  let saving = false;
  let saveMsg = "";

  let form = {};

  onMount(async () => {
    try {
      const [cfg, mdl] = await Promise.all([getConfig(), getModels()]);
      config.set(cfg);
      form = { ...cfg };
      models = mdl;
    } catch (e) {
      console.error("Failed to load config:", e);
    }
  });

  async function save() {
    saving = true;
    saveMsg = "";
    try {
      const updated = await putConfig(form);
      config.set(updated);
      form = { ...updated };
      pipelineState.set("configured");
      saveMsg = "Pipeline configured";
    } catch (e) {
      saveMsg = "Error: " + e.message;
    }
    saving = false;
  }

  async function handleBibUpload() {
    if (!bibFile) return;
    try {
      const res = await uploadBibList(bibFile);
      bibCount = res.count;
      form.bib_set_path = res.path;
    } catch (e) {
      console.error("Upload failed:", e);
    }
  }
</script>

<div class="setup">
  <div class="page-header">
    <h2>Pipeline Configuration</h2>
    <p class="page-desc">Set up camera source, models, and detection parameters before starting</p>
  </div>

  <div class="form-grid">
    <div class="section">
      <div class="section-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="section-icon">
          <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        <h3>Camera Source</h3>
      </div>
      <div class="field">
        <label for="source">Source</label>
        <select id="source" bind:value={form.source}>
          <option value="csi">CSI (Jetson)</option>
          <option value="usb">USB Webcam</option>
          <option value="file">Video File</option>
        </select>
      </div>
      {#if form.source === "file"}
        <div class="field">
          <label for="vpath">Video Path</label>
          <input id="vpath" type="text" bind:value={form.video_path} placeholder="path/to/video.mp4" />
        </div>
      {:else}
        <div class="field">
          <label for="devid">Device Index</label>
          <input id="devid" type="number" bind:value={form.device_index} min="0" max="10" />
        </div>
      {/if}
      <div class="field-row">
        <div class="field">
          <label for="w">Width</label>
          <input id="w" type="number" bind:value={form.width} />
        </div>
        <div class="field">
          <label for="h">Height</label>
          <input id="h" type="number" bind:value={form.height} />
        </div>
        <div class="field">
          <label for="fps">FPS</label>
          <input id="fps" type="number" bind:value={form.fps} />
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="section-icon">
          <path d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
        </svg>
        <h3>Models</h3>
      </div>
      <div class="field">
        <label for="detector">Detector</label>
        <input id="detector" type="text" bind:value={form.detector_path} />
        {#if models.detectors.length}
          <span class="hint">Available: {models.detectors.join(", ")}</span>
        {/if}
      </div>
      <div class="field">
        <label for="ocrmodel">OCR Model</label>
        <select id="ocrmodel" bind:value={form.ocr_model}>
          <option value="parseq">PARSeq</option>
          <option value="crnn">CRNN</option>
        </select>
      </div>
      <div class="field">
        <label for="ocrbackend">OCR Backend</label>
        <select id="ocrbackend" bind:value={form.ocr_backend}>
          <option value="pytorch">PyTorch</option>
          <option value="onnx">ONNX</option>
          <option value="tensorrt">TensorRT</option>
        </select>
      </div>
      <div class="field">
        <label for="pose">Pose Model</label>
        <input id="pose" type="text" bind:value={form.pose_model} />
      </div>
    </div>

    <div class="section">
      <div class="section-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="section-icon">
          <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <h3>Detection</h3>
      </div>
      <div class="field">
        <label for="confthresh">
          Detection Threshold
          <span class="range-val">{(form.conf_threshold || 0.25).toFixed(2)}</span>
        </label>
        <input id="confthresh" type="range" bind:value={form.conf_threshold} min="0.1" max="0.9" step="0.05" />
      </div>
      <div class="field">
        <label for="ocrthresh">
          OCR Threshold
          <span class="range-val">{(form.ocr_conf_threshold || 0.5).toFixed(2)}</span>
        </label>
        <input id="ocrthresh" type="range" bind:value={form.ocr_conf_threshold} min="0.1" max="0.9" step="0.05" />
      </div>
      <div class="field-row">
        <div class="field">
          <label for="placement">Placement</label>
          <select id="placement" bind:value={form.placement}>
            <option value="left">Left</option>
            <option value="center">Center</option>
            <option value="right">Right</option>
          </select>
        </div>
        <div class="field">
          <label for="crossmode">Crossing Mode</label>
          <select id="crossmode" bind:value={form.crossing_mode}>
            <option value="zone">Zone</option>
            <option value="line">Line</option>
          </select>
        </div>
      </div>
      {#if form.crossing_mode === "line"}
        <div class="field">
          <label for="tline">Timing Line (x1,y1,x2,y2)</label>
          <input id="tline" type="text" bind:value={form.timing_line} placeholder="0.5,0.0,0.5,1.0" />
        </div>
      {/if}
      <div class="field-row">
        <div class="field">
          <label for="crossdir">Direction</label>
          <select id="crossdir" bind:value={form.crossing_direction}>
            <option value="any">Any</option>
            <option value="left_to_right">Left to Right</option>
            <option value="right_to_left">Right to Left</option>
          </select>
        </div>
        <div class="field">
          <label for="stride">Stride</label>
          <input id="stride" type="number" bind:value={form.stride} min="1" max="4" />
        </div>
      </div>
      <label class="checkbox-field">
        <input type="checkbox" bind:checked={form.record} />
        <span class="checkbox-label">Record raw video</span>
      </label>
    </div>

    <div class="section">
      <div class="section-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="section-icon">
          <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        <h3>Bib List</h3>
      </div>
      <div class="field">
        <label for="bibupload">Upload bib list (one bib per line)</label>
        <div class="file-upload">
          <input
            id="bibupload"
            type="file"
            accept=".txt,.csv"
            on:change={(e) => {
              bibFile = e.target.files[0];
              handleBibUpload();
            }}
          />
        </div>
      </div>
      {#if bibCount !== null}
        <div class="info-badge">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="info-icon">
            <path d="M5 13l4 4L19 7" />
          </svg>
          Loaded {bibCount} bibs
        </div>
      {/if}
      <div class="field">
        <label for="bibrange">Or set bib range</label>
        <input id="bibrange" type="text" bind:value={form.bib_range} placeholder="1-3000" />
      </div>
      {#if form.bib_set_path}
        <p class="file-path">{form.bib_set_path}</p>
      {/if}
    </div>
  </div>

  <div class="actions">
    <button class="btn-save" on:click={save} disabled={saving}>
      {#if saving}
        <span class="saving-spinner"></span>
        Saving...
      {:else}
        Save & Configure
      {/if}
    </button>
    {#if saveMsg}
      <span class="save-msg" class:error={saveMsg.startsWith("Error")}>{saveMsg}</span>
    {/if}
  </div>
</div>

<style>
  .setup {
    max-width: 920px;
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

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
  }

  @media (max-width: 700px) {
    .form-grid { grid-template-columns: 1fr; }
  }

  .section {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .section-icon {
    width: 16px;
    height: 16px;
    color: var(--accent);
    opacity: 0.7;
  }

  h3 {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
  }

  .field {
    margin-bottom: 0.65rem;
  }

  label {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    margin-bottom: 0.3rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
  }

  input[type="text"],
  input[type="number"],
  select {
    display: block;
    width: 100%;
    padding: 0.45rem 0.6rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 0.82rem;
    font-family: var(--mono);
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  input[type="text"]:focus,
  input[type="number"]:focus,
  select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-dim);
  }

  input[type="text"]::placeholder {
    color: var(--text-muted);
    font-family: "DM Sans", system-ui, sans-serif;
  }

  select {
    cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23484f58' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.5rem center;
    padding-right: 1.5rem;
  }

  input[type="range"] {
    display: block;
    width: 100%;
    margin-top: 0.15rem;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .range-val {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent);
  }

  .field-row {
    display: flex;
    gap: 0.5rem;
  }
  .field-row .field { flex: 1; }

  .hint {
    display: block;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
    font-weight: 400;
  }

  .info-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    color: var(--success-bright);
    background: rgba(46, 160, 67, 0.1);
    border: 1px solid rgba(46, 160, 67, 0.2);
    padding: 0.3rem 0.6rem;
    border-radius: var(--radius-sm);
    margin-bottom: 0.65rem;
  }
  .info-icon {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
  }

  .file-path {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    word-break: break-all;
    margin-top: 0.25rem;
  }

  .checkbox-field {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    font-size: 0.82rem;
    margin-top: 0.25rem;
  }
  .checkbox-field input {
    width: auto;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .checkbox-label {
    color: var(--text-secondary);
    font-weight: 500;
  }

  .file-upload input[type="file"] {
    display: block;
    font-size: 0.78rem;
    color: var(--text-secondary);
  }
  .file-upload input[type="file"]::file-selector-button {
    font-family: "DM Sans", system-ui, sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.35rem 0.7rem;
    margin-right: 0.6rem;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    background: var(--bg-elevated);
    color: var(--text-primary);
    cursor: pointer;
    transition: background 0.15s;
  }
  .file-upload input[type="file"]::file-selector-button:hover {
    background: var(--bg-surface);
  }

  .actions {
    margin-top: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .btn-save {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.5rem;
    background: var(--success);
    color: #fff;
    border: 1px solid rgba(46, 160, 67, 0.4);
    border-radius: var(--radius-md);
    font-family: "DM Sans", system-ui, sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-save:hover:not(:disabled) {
    background: #2fb74d;
    box-shadow: 0 0 12px rgba(46, 160, 67, 0.3);
  }
  .btn-save:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .saving-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  .save-msg {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--success-bright);
  }
  .save-msg.error {
    color: var(--danger);
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
