<script>
  import { onMount } from "svelte";
  import {
    getNetworkStatus,
    scanWifi,
    connectWifi,
    disconnectWifi,
  } from "../api.js";

  let interfaces = [];
  let wifiNetworks = [];
  let loading = true;
  let scanning = false;
  let connectingSsid = null;
  let selectedSsid = null;
  let password = "";
  let error = "";
  let success = "";

  onMount(async () => {
    await refreshStatus();
    loading = false;
  });

  async function refreshStatus() {
    try {
      interfaces = await getNetworkStatus();
    } catch (e) {
      console.error("Failed to load network status:", e);
    }
  }

  async function handleScan() {
    scanning = true;
    error = "";
    success = "";
    try {
      wifiNetworks = await scanWifi();
    } catch (e) {
      error = "Scan failed: " + e.message;
    }
    scanning = false;
  }

  async function handleConnect(ssid) {
    connectingSsid = ssid;
    error = "";
    success = "";
    try {
      await connectWifi(ssid, password);
      success = `Connected to ${ssid}`;
      selectedSsid = null;
      password = "";
      await Promise.all([refreshStatus(), handleScan()]);
    } catch (e) {
      error = "Connect failed: " + e.message;
    }
    connectingSsid = null;
  }

  async function handleDisconnect(device) {
    error = "";
    success = "";
    try {
      await disconnectWifi(device);
      success = `Disconnected ${device}`;
      await refreshStatus();
    } catch (e) {
      error = "Disconnect failed: " + e.message;
    }
  }

  function selectNetwork(ssid) {
    if (selectedSsid === ssid) {
      selectedSsid = null;
      password = "";
    } else {
      selectedSsid = ssid;
      password = "";
    }
  }

  function signalBars(signal) {
    if (signal >= 75) return 4;
    if (signal >= 50) return 3;
    if (signal >= 25) return 2;
    return 1;
  }

  function stateColor(state) {
    if (state === "connected") return "var(--success-bright)";
    if (state === "connecting") return "var(--warning)";
    return "var(--text-muted)";
  }

  function typeLabel(type) {
    if (type === "wifi") return "WiFi";
    if (type === "ethernet") return "Ethernet";
    return type;
  }
</script>

<div class="network">
  <div class="page-header">
    <h2>Network</h2>
    <p class="page-desc">Manage network interfaces and WiFi connections</p>
  </div>

  {#if error}
    <div class="msg msg-error">{error}</div>
  {/if}
  {#if success}
    <div class="msg msg-success">{success}</div>
  {/if}

  <div class="sections">
    <!-- Interface Status -->
    <div class="section">
      <div class="section-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="section-icon">
          <path d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
        </svg>
        <h3>Interfaces</h3>
      </div>

      {#if loading}
        <div class="placeholder">Loading...</div>
      {:else if interfaces.length === 0}
        <div class="placeholder">No interfaces found</div>
      {:else}
        <div class="iface-grid">
          {#each interfaces as iface}
            <div class="iface-card">
              <div class="iface-top">
                <span class="iface-device">{iface.device}</span>
                <span class="type-badge">{typeLabel(iface.type)}</span>
              </div>
              <div class="iface-status">
                <span class="state-dot" style="background: {stateColor(iface.state)}"></span>
                <span class="state-label">{iface.connection || iface.state}</span>
              </div>
              {#if iface.ips.length > 0}
                <div class="iface-ips">
                  {#each iface.ips as ip}
                    <span class="ip-addr">{ip}</span>
                  {/each}
                </div>
              {/if}
              {#if iface.state === "connected"}
                <button class="btn-sm btn-disconnect" on:click={() => handleDisconnect(iface.device)}>
                  Disconnect
                </button>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- WiFi Networks -->
    <div class="section">
      <div class="section-header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="section-icon">
          <path d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01M4.93 12.93a10 10 0 0114.14 0M1.394 9.393a14 14 0 0121.213 0" />
        </svg>
        <h3>WiFi Networks</h3>
        <button class="btn-sm btn-scan" on:click={handleScan} disabled={scanning}>
          {#if scanning}
            <span class="spinner"></span>
            Scanning...
          {:else}
            Scan
          {/if}
        </button>
      </div>

      {#if wifiNetworks.length === 0 && !scanning}
        <div class="placeholder">Press Scan to find WiFi networks</div>
      {:else}
        <div class="wifi-list">
          {#each wifiNetworks as net}
            <button
              class="wifi-row"
              class:active={selectedSsid === net.ssid}
              class:connected={net.in_use}
              on:click={() => selectNetwork(net.ssid)}
            >
              <div class="signal-bars" title="{net.signal}%">
                {#each [1, 2, 3, 4] as bar}
                  <div class="bar" class:active={bar <= signalBars(net.signal)}></div>
                {/each}
              </div>
              <span class="wifi-ssid">{net.ssid}</span>
              {#if net.security && net.security !== "--"}
                <span class="security-badge">{net.security}</span>
              {/if}
              {#if net.in_use}
                <span class="connected-badge">Connected</span>
              {/if}
            </button>

            {#if selectedSsid === net.ssid && !net.in_use}
              <div class="connect-form">
                <input
                  type="password"
                  bind:value={password}
                  placeholder="Password"
                  on:keydown={(e) => e.key === "Enter" && handleConnect(net.ssid)}
                />
                <button
                  class="btn-sm btn-connect"
                  on:click={() => handleConnect(net.ssid)}
                  disabled={connectingSsid === net.ssid}
                >
                  {#if connectingSsid === net.ssid}
                    <span class="spinner"></span>
                  {:else}
                    Connect
                  {/if}
                </button>
              </div>
            {/if}
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .network {
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

  .sections {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
  }

  @media (max-width: 700px) {
    .sections { grid-template-columns: 1fr; }
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

  .placeholder {
    font-size: 0.82rem;
    color: var(--text-muted);
    padding: 1rem 0;
    text-align: center;
  }

  .msg {
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius-sm);
    margin-bottom: 0.85rem;
  }
  .msg-error {
    color: var(--danger);
    background: var(--danger-dim);
    border: 1px solid rgba(218, 54, 51, 0.25);
  }
  .msg-success {
    color: var(--success-bright);
    background: rgba(46, 160, 67, 0.1);
    border: 1px solid rgba(46, 160, 67, 0.2);
  }

  /* Interface cards */
  .iface-grid {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .iface-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.65rem 0.75rem;
  }

  .iface-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
  }

  .iface-device {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .type-badge {
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    background: var(--bg-surface);
    padding: 0.15rem 0.45rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
  }

  .iface-status {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.3rem;
  }

  .state-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .state-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
  }

  .iface-ips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-bottom: 0.35rem;
  }

  .ip-addr {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--accent);
    background: var(--accent-dim);
    padding: 0.1rem 0.45rem;
    border-radius: var(--radius-sm);
  }

  /* WiFi list */
  .wifi-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .wifi-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.5rem 0.6rem;
    background: var(--bg-elevated);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background 0.12s, border-color 0.12s;
    text-align: left;
    font-family: inherit;
    color: inherit;
  }

  .wifi-row:hover {
    background: var(--bg-base);
  }

  .wifi-row.active {
    border-color: var(--accent);
    background: var(--bg-base);
  }

  .wifi-row.connected {
    border-color: rgba(46, 160, 67, 0.3);
  }

  .wifi-ssid {
    flex: 1;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .security-badge {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-surface);
    padding: 0.1rem 0.35rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
    white-space: nowrap;
  }

  .connected-badge {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--success-bright);
  }

  /* Signal bars */
  .signal-bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 14px;
    flex-shrink: 0;
  }

  .signal-bars .bar {
    width: 3px;
    border-radius: 1px;
    background: var(--border-default);
    transition: background 0.15s;
  }

  .signal-bars .bar:nth-child(1) { height: 4px; }
  .signal-bars .bar:nth-child(2) { height: 7px; }
  .signal-bars .bar:nth-child(3) { height: 10px; }
  .signal-bars .bar:nth-child(4) { height: 14px; }

  .signal-bars .bar.active {
    background: var(--accent);
  }

  /* Connect form */
  .connect-form {
    display: flex;
    gap: 0.4rem;
    padding: 0.5rem 0.6rem;
    background: var(--bg-base);
    border-radius: var(--radius-sm);
    margin-top: 1px;
  }

  .connect-form input {
    flex: 1;
    padding: 0.4rem 0.55rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 0.82rem;
    font-family: var(--mono);
  }

  .connect-form input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-dim);
  }

  .connect-form input::placeholder {
    color: var(--text-muted);
    font-family: "DM Sans", system-ui, sans-serif;
  }

  /* Buttons */
  .btn-sm {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.65rem;
    font-family: "DM Sans", system-ui, sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.15s;
    border: 1px solid;
    white-space: nowrap;
  }

  .btn-sm:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-scan {
    margin-left: auto;
    color: var(--accent);
    background: var(--accent-dim);
    border-color: rgba(0, 180, 216, 0.2);
  }

  .btn-scan:hover:not(:disabled) {
    background: rgba(0, 180, 216, 0.2);
  }

  .btn-connect {
    color: #fff;
    background: var(--success);
    border-color: rgba(46, 160, 67, 0.4);
  }

  .btn-connect:hover:not(:disabled) {
    background: #2fb74d;
  }

  .btn-disconnect {
    color: var(--danger);
    background: var(--danger-dim);
    border-color: rgba(218, 54, 51, 0.25);
    margin-top: 0.35rem;
  }

  .btn-disconnect:hover {
    background: rgba(218, 54, 51, 0.2);
  }

  .spinner {
    width: 12px;
    height: 12px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
