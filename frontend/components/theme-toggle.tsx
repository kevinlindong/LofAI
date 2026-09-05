"use client"

import { useEffect, useState } from "react"
import { DotGlyph } from "@/components/dot-glyph"

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(true)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    // the inline script in the document head already applied the theme; read
    // back what it decided rather than deciding again
    setIsDark(document.documentElement.classList.contains("dark"))
  }, [])

  const toggle = () => {
    const next = !isDark
    setIsDark(next)
    localStorage.setItem("darkMode", String(next))
    document.documentElement.classList.toggle("dark", next)
  }

  return (
    <button
      type="button"
      onClick={toggle}
      className="key theme-control"
      style={{ opacity: mounted ? 1 : 0 }}
      aria-label={isDark ? "Use light theme" : "Use dark theme"}
    >
      <DotGlyph name={isDark ? "sun" : "moon"} dot={2} />
      <span className="hidden sm:inline">{isDark ? "light" : "dark"}</span>
    </button>
  )
}

export default ThemeToggle
