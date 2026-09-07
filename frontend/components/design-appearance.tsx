"use client"

import { useEffect, useLayoutEffect } from "react"
import { usePathname } from "next/navigation"
import { applyDesignAppearance, DESIGNS, type DesignTone } from "@/lib/designs"

const useBrowserLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect

// Appearance belongs to the active URL, even when the router retains an old
// page in its cache. Page-level cleanup alone cannot enforce that boundary.
export function DesignAppearance() {
  const pathname = usePathname()

  useBrowserLayoutEffect(() => {
    const design = DESIGNS.find((entry) => pathname?.replace(/\/$/, "") === `/${entry.id}`)
    let tone: DesignTone = "default"
    if (design && design.id !== "1") {
      try {
        const saved = JSON.parse(localStorage.getItem(`lofai.design.${design.id}`) || "{}")
        if (saved && ["default", "alternate", "mono"].includes(saved.tone)) tone = saved.tone
      } catch { /* The default remains usable without browser storage. */ }
    }
    applyDesignAppearance(design, tone)
  }, [pathname])

  return null
}
