<script>
  import { onMount } from "svelte";
  import Setup from "./lib/components/Setup.svelte";
  import Live from "./lib/components/Live.svelte";
  import Review from "./lib/components/Review.svelte";
  import Network from "./lib/components/Network.svelte";

  let page = "live";

  function updatePage() {
    const hash = location.hash.replace("#/", "").replace("#", "") || "live";
    if (["setup", "live", "review", "network"].includes(hash)) {
      page = hash;
    } else {
      page = "live";
    }
  }

  onMount(() => {
    updatePage();
    window.addEventListener("hashchange", updatePage);
    return () => window.removeEventListener("hashchange", updatePage);
  });

  const navItems = [
    { id: "setup", label: "Setup", icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065zM15 12a3 3 0 11-6 0 3 3 0 016 0z" },
    { id: "live", label: "Live", icon: "M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" },
    { id: "review", label: "Review", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" },
    { id: "network", label: "Network", icon: "M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01M4.93 12.93a10 10 0 0114.14 0M1.394 9.393a14 14 0 0121.213 0" },
  ];
</script>

<div class="app">
  <nav>
    <a href="#/" class="logo">
      <span class="logo-mark"></span>
      <span class="logo-text">PointCam</span>
    </a>
    <div class="nav-links">
      {#each navItems as item}
        <a href="#/{item.id}" class:active={page === item.id}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d={item.icon} />
          </svg>
          {item.label}
        </a>
      {/each}
    </div>
  </nav>

  <main>
    {#if page === "setup"}
      <Setup />
    {:else if page === "live"}
      <Live />
    {:else if page === "review"}
      <Review />
    {:else if page === "network"}
      <Network />
    {/if}
  </main>
</div>

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  :global(body) {
    font-family: "DM Sans", system-ui, sans-serif;
    background: var(--bg-base);
    color: var(--text-primary);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  :global(:root) {
    --bg-base: #080b12;
    --bg-surface: #0d1117;
    --bg-elevated: #151b26;
    --bg-overlay: rgba(13, 17, 23, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-default: rgba(255, 255, 255, 0.1);
    --border-focus: #00b4d8;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --accent: #00b4d8;
    --accent-dim: rgba(0, 180, 216, 0.15);
    --accent-glow: rgba(0, 180, 216, 0.3);
    --success: #2ea043;
    --success-bright: #3fb950;
    --warning: #d29922;
    --danger: #da3633;
    --danger-dim: rgba(218, 54, 51, 0.15);
    --mono: "JetBrains Mono", "Fira Code", monospace;
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
  }
  :global(::selection) {
    background: var(--accent-dim);
    color: var(--accent);
  }
  :global(::-webkit-scrollbar) {
    width: 6px;
    height: 6px;
  }
  :global(::-webkit-scrollbar-track) {
    background: transparent;
  }
  :global(::-webkit-scrollbar-thumb) {
    background: var(--border-default);
    border-radius: 3px;
  }
  :global(::-webkit-scrollbar-thumb:hover) {
    background: var(--text-muted);
  }

  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  nav {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 0 1.25rem;
    height: 52px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-subtle);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    text-decoration: none;
    flex-shrink: 0;
  }
  .logo-mark {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow), 0 0 20px rgba(0, 180, 216, 0.15);
  }
  .logo-text {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-primary);
    letter-spacing: -0.02em;
  }

  .nav-links {
    display: flex;
    gap: 2px;
  }
  .nav-links a {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--text-secondary);
    text-decoration: none;
    padding: 0.45rem 0.85rem;
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: color 0.15s, background 0.15s;
  }
  .nav-links a svg {
    width: 16px;
    height: 16px;
    opacity: 0.6;
    transition: opacity 0.15s;
  }
  .nav-links a:hover {
    color: var(--text-primary);
    background: var(--bg-elevated);
  }
  .nav-links a:hover svg {
    opacity: 0.85;
  }
  .nav-links a.active {
    color: var(--text-primary);
    background: var(--bg-elevated);
  }
  .nav-links a.active svg {
    opacity: 1;
    color: var(--accent);
  }

  main {
    flex: 1;
    padding: 1.25rem;
    max-width: 1440px;
    width: 100%;
    margin: 0 auto;
  }
</style>
