import "./globals.css"
import localFont from "next/font/local"
import type React from "react"

// The matrix face is intentionally limited to small moments of identity. The
// rest of the interface uses the system sans stack declared in globals.css.
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

// Pick the theme before first paint so the ambient field and surfaces agree.
const NO_FLASH = `
try {
  var s = localStorage.getItem("darkMode");
  var dark = s === null ? true : s === "true";
  document.documentElement.classList.toggle("dark", dark);
} catch (e) {
  document.documentElement.classList.add("dark");
}`

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={dotMatrix.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: NO_FLASH }} />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
