import "./globals.css"
import localFont from "next/font/local"
import type React from "react"

// the interface is one continuous dot matrix display, so the type is too.
// this face is a real 5x7 matrix - round dots, no strokes - which is why
// everything is set with wide tracking: the glyphs need air to resolve.
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

// pick the theme before first paint - a panel that flashes white and then
// goes black is the one thing that breaks the illusion of hardware
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
