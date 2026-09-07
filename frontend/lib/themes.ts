export const THEMES = [
  { id: "light", name: "Light", description: "Warm paper & coral", dark: false },
  { id: "dark", name: "Dark", description: "Charcoal & soft peach", dark: true },
  { id: "ocean", name: "Ocean", description: "Deep blue & sea glass", dark: true },
  { id: "dusk", name: "Dusk", description: "Violet & lavender", dark: true },
  { id: "forest", name: "Forest", description: "Moss & golden light", dark: true },
  { id: "rose", name: "Rose", description: "Blush & dusty pink", dark: false },
] as const

export type ThemeId = (typeof THEMES)[number]["id"]
export const THEME_STORAGE_KEY = "lofai.theme"

export function getActiveTheme(): ThemeId {
  return THEMES.find((theme) => theme.id === document.documentElement.dataset.theme)?.id ?? "dark"
}

export function applyTheme(id: ThemeId): boolean {
  const theme = THEMES.find((entry) => entry.id === id)!
  const root = document.documentElement
  root.dataset.theme = id
  root.classList.remove(...THEMES.map((entry) => `theme-${entry.id}`))
  root.classList.add(`theme-${id}`)
  root.classList.toggle("dark", theme.dark)

  try {
    localStorage.setItem(THEME_STORAGE_KEY, id)
    return true
  } catch {
    // The palette can still change when browser storage is unavailable.
    return false
  }
}

// Shared with the document head so saved palettes are applied before paint.
// Existing light/dark preferences remain valid when upgrading to color themes.
export const THEME_INIT_SCRIPT = `
(function () {
  var themes = ${JSON.stringify(THEMES)};
  var id = "dark";
  try {
    var saved = localStorage.getItem(${JSON.stringify(THEME_STORAGE_KEY)});
    if (themes.some(function (theme) { return theme.id === saved; })) id = saved;
    else if (localStorage.getItem("darkMode") === "false") id = "light";
  } catch (e) {}
  var theme = themes.find(function (entry) { return entry.id === id; });
  var root = document.documentElement;
  root.dataset.theme = id;
  root.classList.add("theme-" + id);
  root.classList.toggle("dark", theme.dark);
})();`
