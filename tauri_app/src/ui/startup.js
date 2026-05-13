export function createStartupLoader({ loader, percent, fill }) {
  function set(value) {
    const pct = Math.max(0, Math.min(100, Math.round(value)));
    if (percent) percent.textContent = `${pct}%`;
    if (fill) {
      fill.style.animation = 'none';
      fill.style.transform = 'none';
      fill.style.width = `${pct}%`;
    }
  }

  function hide() {
    set(100);
    window.setTimeout(() => {
      loader?.classList.add('done');
      window.setTimeout(() => loader?.remove(), 420);
    }, 180);
  }

  return { set, hide };
}
