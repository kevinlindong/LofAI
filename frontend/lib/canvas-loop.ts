// Keep motion at 60fps without doing twice the painting on a 120Hz display.
// Hidden/offscreen canvases stop entirely; resuming starts with a small dt.
export function canvasLoop(
  element: Element,
  draw: (now: number, dt: number) => void,
  still: () => void,
) {
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)")
  const interval = 1000 / 60
  let visible = true
  let disposed = false
  let raf = 0
  let previous = 0
  let due = 0

  const tick = (now: number) => {
    if (now + 0.5 >= due) {
      draw(now, previous ? Math.min(0.05, (now - previous) / 1000) : 1 / 60)
      previous = now
      due = Math.max(due + interval, now)
    }
    raf = requestAnimationFrame(tick)
  }

  const sync = () => {
    cancelAnimationFrame(raf)
    raf = 0
    previous = 0
    due = 0
    if (disposed || document.hidden || !visible) return
    if (motion.matches) still()
    else raf = requestAnimationFrame(tick)
  }

  const observer = new IntersectionObserver(([entry]) => {
    if (visible === entry.isIntersecting) return
    visible = entry.isIntersecting
    sync()
  })
  observer.observe(element)
  document.addEventListener("visibilitychange", sync)
  motion.addEventListener("change", sync)
  sync()

  return {
    redraw() {
      if (!disposed && !document.hidden && visible && motion.matches) still()
    },
    dispose() {
      disposed = true
      cancelAnimationFrame(raf)
      observer.disconnect()
      document.removeEventListener("visibilitychange", sync)
      motion.removeEventListener("change", sync)
    },
  }
}
