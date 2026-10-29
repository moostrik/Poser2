// Injected only into the Edge app window (?app=1): tie the window's lifetime to the server's on every
// exit path. A dropped socket alone does not mean the server died: a long settings callback can stall
// the event loop past the heartbeat timeout, so on disconnect the script probes the server. A dead one
// refuses the fetch instantly (close); a stalled one leaves it hanging (keep waiting, socket.io
// reconnects by itself). Server-side pushes cannot drive the close: the panel's exit button tears the
// app down on the UI event loop, which never flushes a pushed close. The beforeunload guard keeps a page
// reload (self-heal) from closing the window.
// Self-heal: NiceGUI's #esm-fallback banner still present a few seconds in means a page resource failed
// to load; reload rather than sit on a dead page.
(function () {
  let unloading = false;
  window.addEventListener("beforeunload", () => { unloading = true; });

  setTimeout(() => {
    if (document.getElementById("esm-fallback")) window.location.reload();
  }, 4000);

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function serverGone() {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 1500);
    try {
      await fetch(window.location.href, { cache: "no-store", signal: abort.signal });
      return false;                    // any response: alive
    } catch (e) {
      return e.name !== "AbortError";  // instant refusal: gone; abort: stalled, not gone
    } finally {
      clearTimeout(timer);
    }
  }

  async function closeWhenServerGone() {
    while (!unloading && !window.socket.connected) {
      if (await serverGone()) {
        window.close();
        return;
      }
      await sleep(1000);
    }
  }

  const timer = setInterval(() => {
    if (!window.socket) return;
    clearInterval(timer);
    window.socket.on("disconnect", () => { if (!unloading) closeWhenServerGone(); });
  }, 100);
})();
