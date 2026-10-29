// The panel's client-side helpers. Each takes its parameters as arguments; the page calls them with
// the values it formats in (the title, the drawer height, element ids), nothing is formatted into
// this file.
window.panel = (function () {
  // Waits for NiceGUI's socket, then calls `attach` with it: the socket is created after the page's
  // head scripts run.
  function withSocket(attach) {
    const timer = setInterval(function () {
      if (!window.socket) return;
      clearInterval(timer);
      attach(window.socket);
    }, 100);
  }

  // An overlay over the whole page while the server is gone: shown 1.5 s after the socket drops
  // (an external quit, Ctrl-C, stop() from code) and taken away when it is back. The exit button
  // shows it at once (`showShutdownScreen`). `drawerHeight` is a CSS length: the message is
  // centred in the part of the page above the log drawer.
  function installShutdownOverlay(title, drawerHeight) {
    function show() {
      if (document.getElementById("shutdown-overlay")) return;
      const overlay = document.createElement("div");
      overlay.id = "shutdown-overlay";
      overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:9998;"
        + "backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);";
      const box = document.createElement("div");
      box.style.cssText = "position:absolute;left:50%;top:calc((100vh - " + drawerHeight + ") / 2);"
        + "transform:translate(-50%, -50%);text-align:center;color:#e4e4e7";
      if (title) {
        const heading = document.createElement("div");
        heading.style.cssText = "font-size:1.5rem;font-weight:bold;color:#3f3f46;margin-bottom:1rem";
        heading.textContent = title;
        box.appendChild(heading);
      }
      const icon = document.createElement("span");
      icon.className = "material-icons";
      icon.style.cssText = "font-size:64px;color:#71717a;margin-bottom:0.75rem;display:block";
      icon.textContent = "cloud_off";
      box.appendChild(icon);
      const line = document.createElement("div");
      line.style.cssText = "font-size:1rem;font-weight:500;color:#a1a1aa";
      line.textContent = "No connection";
      box.appendChild(line);
      const sub = document.createElement("div");
      sub.style.cssText = "font-size:0.8rem;color:#71717a;margin-top:0.25rem";
      sub.textContent = "Waiting for server…";
      box.appendChild(sub);
      overlay.appendChild(box);
      document.body.appendChild(overlay);
    }
    window.showShutdownScreen = show;
    let pending = null;
    withSocket(function (socket) {
      socket.on("disconnect", function () { pending = setTimeout(show, 1500); });
      socket.on("connect", function () {
        if (pending) { clearTimeout(pending); pending = null; }
        const overlay = document.getElementById("shutdown-overlay");
        if (overlay) overlay.remove();
      });
    });
  }

  // The pinned bar is scaled with a transform, which keeps its layout size: this gives the frame
  // around it the scaled height, and follows the bar when it wraps.
  function watchPinnedFrame() {
    const timer = setInterval(function () {
      const frame = document.querySelector(".poser-pinned-frame");
      const bar = frame && frame.querySelector(".poser-pinned");
      if (!bar) return;
      clearInterval(timer);
      const scale = parseFloat(getComputedStyle(frame).getPropertyValue("--scale"));
      new ResizeObserver(function () {
        frame.style.height = bar.offsetHeight * scale + "px";
      }).observe(bar);
    }, 100);
  }

  function element(id) {
    return document.getElementById("c" + id);
  }

  // The log drawer's handle: dragging it up and down sets the drawer's height.
  function dragHandle(handleId, drawerId) {
    const handle = element(handleId);
    const drawer = element(drawerId);
    if (!handle || !drawer) return;
    let startY = 0, startHeight = 0, dragging = false;
    handle.addEventListener("mousedown", function (e) {
      dragging = true;
      startY = e.clientY;
      startHeight = drawer.offsetHeight;
      e.preventDefault();
    });
    document.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      const height = startHeight + (startY - e.clientY);
      drawer.style.height = Math.max(100, Math.min(window.innerHeight - 60, height)) + "px";
    });
    document.addEventListener("mouseup", function () { dragging = false; });
  }

  // Tells the server whether the log area is scrolled to its bottom (`log_scroll` event), so it
  // keeps following new lines only while the user has not scrolled up.
  function reportScroll(areaId) {
    const area = element(areaId);
    if (!area) return;
    const atBottom = (area.scrollHeight - area.scrollTop - area.clientHeight) < 30;
    emitEvent("log_scroll", { atBottom: atBottom });
  }

  function scrollToBottom(areaId) {
    const area = element(areaId);
    if (area) area.scrollTop = area.scrollHeight;
  }

  return {
    installShutdownOverlay: installShutdownOverlay,
    watchPinnedFrame: watchPinnedFrame,
    dragHandle: dragHandle,
    reportScroll: reportScroll,
    scrollToBottom: scrollToBottom,
  };
})();
