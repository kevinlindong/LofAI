import "./globals.css"
import "./designs.css"
import "./sound-editor.css"
import localFont from "next/font/local"
import type React from "react"
import { THEME_INIT_SCRIPT } from "@/lib/themes"
import { DESIGN_INIT_SCRIPT } from "@/lib/designs"
import { RadioProvider } from "@/components/radio-provider"
import { DesignAppearance } from "@/components/design-appearance"

// The matrix face textures the ambient background. The interface uses the
// system sans stack declared in globals.css.
const dotMatrix = localFont({
  src: "./fonts/dotmatrix.woff2",
  variable: "--font-dot",
  display: "swap",
  fallback: ["Courier New", "ui-monospace", "monospace"],
})

export const metadata = {
  title: "lofAI",
  description: "Endless lofi, generated live, steered while it plays",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={dotMatrix.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT + DESIGN_INIT_SCRIPT }} />
      </head>
      <body suppressHydrationWarning><DesignAppearance /><RadioProvider>{children}</RadioProvider></body>
    </html>
  )
}
