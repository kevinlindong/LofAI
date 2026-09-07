export const DESIGNS = [
  { id: "1", name: "Original", slug: "original", description: "Your familiar space" },
  { id: "2", name: "Sunday", slug: "sunday", description: "The listening room" },
  { id: "3", name: "Form", slug: "form", description: "A space for good work" },
  { id: "4", name: "Signal", slug: "signal", description: "An endless transmission" },
  { id: "5", name: "Afterglow", slug: "afterglow", description: "For the quieter hours" },
] as const

export type Design = (typeof DESIGNS)[number]
export type NewDesign = Exclude<Design, { id: "1" }>
export type DesignTone = "default" | "alternate" | "mono"

export function applyDesignAppearance(design?: Design, tone: DesignTone = "default") {
  const root = document.documentElement
  root.classList.remove(
    ...DESIGNS.map((entry) => `design-${entry.slug}`),
    "design-tone-default", "design-tone-alternate", "design-tone-mono",
  )
  if (design && design.id !== "1") root.classList.add(`design-${design.slug}`, `design-tone-${tone}`)
}

export const DESIGN_TONES: Record<NewDesign["slug"], readonly { id: DesignTone; name: string; color: string }[]> = {
  sunday: [
    { id: "default", name: "Burgundy", color: "#773a43" },
    { id: "alternate", name: "Olive", color: "#555c3b" },
    { id: "mono", name: "Ink", color: "#393835" },
  ],
  form: [
    { id: "default", name: "Cobalt", color: "#254be3" },
    { id: "alternate", name: "Vermilion", color: "#bb432f" },
    { id: "mono", name: "Graphite", color: "#33393e" },
  ],
  signal: [
    { id: "default", name: "Amber", color: "#f9ad62" },
    { id: "alternate", name: "Phosphor", color: "#acdc9b" },
    { id: "mono", name: "Silver", color: "#dadbd4" },
  ],
  afterglow: [
    { id: "default", name: "Iris", color: "#b9a0de" },
    { id: "alternate", name: "Tide", color: "#8ec7c0" },
    { id: "mono", name: "Moon", color: "#c9c8c5" },
  ],
}

// Match the route before first paint, without changing the original palette.
export const DESIGN_INIT_SCRIPT = `
(function () {
  var designs = ${JSON.stringify(DESIGNS)};
  var design = designs.find(function (entry) { return location.pathname.replace(/\\/$/, "") === "/" + entry.id; });
  if (!design || design.id === "1") return;
  var tone = "default";
  try {
    var saved = JSON.parse(localStorage.getItem("lofai.design." + design.id) || "{}");
    if (["default", "alternate", "mono"].indexOf(saved.tone) !== -1) tone = saved.tone;
  } catch (e) {}
  document.documentElement.classList.add("design-" + design.slug, "design-tone-" + tone);
})();`
