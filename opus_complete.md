# opus_complete

Orchestration record. Two passes: the first put the ink in the wrong place and
built the wrong connector, the second moved it and replaced the connector with
the standard one. Both are written down, because the corrections are the useful
part.

## What was asked

1. The background is lagging the site. Make it minimal and cheap. It does not
   have to use the dot matrix or metaballs at all — subtle ASCII highlights,
   something chill that fits the lofi vibe, ASCII art set in the ENDLESS font.
   It should still be reactive to the cursor the way the old one was.
2. The metaball flow belongs in the **audio visualiser** and the **cat**, on
   their dot matrices, and it has to look like the reference: two dots joined by
   a smooth concave waist, one poured shape rather than beads on a string.
3. The cells can stay dots. They do not have to be perfect circles — a rounded
   square or anything between is allowed.

## Pass one, and what was wrong with it

Pass one put the ink on the page background and hand-rolled the connector. Four
mistakes, all worth keeping on the record:

- **Cost.** A full-viewport lattice sampled twice per dot per frame, three colour
  fields, a path per colour, all of it behind every translucent card on the page.
  That was the lag, and none of that work survived being covered up.
- **Colour-split fields.** Each ink colour had its own field and a dot only
  joined a neighbour its own colour also claimed, so the field broke into
  scattered beads.
- **A bridge threshold above the dot threshold.** Dots wetted before bridges
  appeared and the neck was capped thinner than the dots it joined. That is
  beads-on-a-string by construction: ink does not work that way, and if two cells
  are wet and adjacent the ink between them is already there.
- **A hand-placed control point.** This was the real one. Pass one drew the neck
  as two cubics with the controls pushed inward by eye. That gives *a* pinch, but
  not the shape in the reference — because the reference's membrane leaves each
  circle along that circle's **tangent**, and a guessed control point does not.
  Tangency is the whole difference between liquid and a shape glued on.

## Pass two: use the standard algorithm

Rather than tune the guess, look for the thing that does this exactly. It exists
and it is well travelled:

- **Hiroyuki Sato's metaball** — <http://shspage.com/aijs/en/#metaball>
- **kynd's Paper.js port** — <http://paperjs.org/examples/meta-balls>, the
  `metaball(ball1, ball2, v, handle_len_rate, maxDistance)` everyone copies
- **Varun Vachhar's derivation** — <https://varun.ca/metaballs/>, which is where
  the geometry is actually explained

Also surveyed and rejected, with reasons, because they are the obvious
alternatives and someone will ask:

| Option | Why not |
|---|---|
| SVG gooey filter (`feGaussianBlur` + `feColorMatrix`), `gooey-react` | Blurs. The whole point of this panel is honestly quantised edges, and a threshold-on-blur softens every one of them. Also a per-pixel filter pass, which is the cost we were removing |
| CSS `filter: blur() contrast()` | Same blur objection, and it cannot be scoped to one shade of a palette |
| Marching squares over the field (`d3-contour`) | Gives a correct iso-surface but throws the dot matrix away — the output is one smooth contour, not dots that fuse |
| SDF smooth-minimum (`smin`) in a shader | Correct and beautiful, but it is a WebGL context and a per-pixel pass for something that is a few hundred bezier segments |

Sato's construction is the right shape *and* the cheap one: it is pure path
geometry, so it composes with the existing canvas, respects the palette, and adds
no per-pixel work at all.

### How it works, in one paragraph

The widest a membrane between two circles can be is bounded by their two common
external tangents, and the angle from the centre line to a tangent point is
`acos((r1 - r2) / d)`. `spread` (Sato's `v`, 0.5) is how much of that maximum
this membrane takes. The four points where the membrane meets the circles go at
those angles, and each gets a bezier handle turned a quarter turn off its own
radius — which is to say, laid along the circle's tangent there. Handle length
(`handleSize`, Sato's `handle_len_rate` = 2.4) falls off as the circles separate,
so a membrane thins and lets go rather than snapping. When the circles overlap,
`u1`/`u2` open the spread out by the overlap half-angle from the law of cosines,
so the membrane stays outside the lens they already share, and the whole thing
collapses cleanly once one circle contains the other.

Two deliberate departures from upstream, both noted in the source:

- Upstream draws the connector *plus* the far arc of the second circle, because
  upstream is not also drawing the circles. Here the circles are already in the
  path, so the membrane closes across each circle's own chord instead — a chord
  between two points on a circle is inside it, so the union is identical and
  there is no arc sweep flag to get wrong.
- It is walked `p2 → p4 → p3 → p1` rather than Sato's `p1 → p3 → p4 → p2`. Same
  curve, but this direction winds the same way `arc` sweeps. Under the nonzero
  fill rule two subpaths that wind oppositely **cancel** where they overlap,
  which would punch a hole through every join instead of filling it. There is a
  test for exactly this.

## Work items

### 1. Background: delete the expensive one, add a reactive cheap one

- Deleted `components/metaball-field.tsx` and `components/dot-ambience.tsx` — the
  latter was a *second* full-viewport metaball canvas, already unreferenced;
  leaving it would have left the expensive pattern in the tree as an example.
- Added `components/ascii-ambience.tsx`: six large ASCII marks in ENDLESS plus a
  glyph cluster that trails the pointer.
- **Cursor reactivity without a frame loop.** One passive `pointermove` handler
  stores two floats and schedules a single rAF that writes four custom properties
  — and it is only ever scheduled while the pointer is actually moving, so a
  still cursor costs nothing. Everything downstream is CSS: each mark leans by
  `translate: calc(var(--ax) * var(--depth))` with its own depth for parallax,
  and a long `transition` on `translate` supplies the lag and the easing. The
  trail is positioned the same way from `--px`/`--py`.
- `translate` and not `transform`, deliberately: it is its own property, so it
  composes with the drift keyframes' `transform` instead of being overwritten by
  it. That is what lets one element both drift and follow the cursor with no
  wrapper element per mark.
- `prefers-reduced-motion` returns before installing the listener, and the CSS
  stops the drift and hides the trail as well.

### 2. The renderer

`lib/ink-render.ts`. Cells are circles again — dots, as asked, and as Sato's
construction requires. A cell's radius runs from `minRadius` to `maxRadius`; at
half the pitch two neighbours are exactly tangent and past it they overlap, which
is what lets a solid region fuse rather than tile. Cells fuse only within their
own shade, so the cat's one-cell rim stays an outline around the fur instead of
melting into it. Diagonal membranes only where the L-shaped route is not already
ink. Everything for one colour is one path filled once with the nonzero rule.

### 3 & 4. Wire the visualiser and the cat

The visualiser turns each dot's field value into a continuous fill, so quiet ink
is honestly separate beads and joining up is something the music does. Neighbour
tables are built once in `layout()` because that lattice is sparse — it has a
hole punched in it for the transport button. `pet-scene.ts` exports `INK`, per-cell
fill read off the same field the silhouette is cut from; shade 0 (unlit panel,
eye sockets) draws as small unfused dots.

## The defect the visual check caught

`test/ink-preview.mjs` runs the real renderer against a context that emits SVG
instead of rasterising, so the output can be *looked at*. That is what caught the
one thing no amount of unit testing would have: four circles at the corners of a
lattice square cover its edges — the membranes see to that — but not its middle,
and the pinhole left behind read as a perforated body rather than poured ink.

The fix is targeted: fill the square joining the four centres. Its corners are
the centres, deep inside their own circles, and its edges lie along the
centre-to-centre axes, which every membrane straddles — so it can only ever add
area already inside the union, and never changes the silhouette. It is gated on
all four corners being wet **and** joined the whole way round, since one open
side would let a plug show.

Regenerate the preview with:

```bash
cd frontend && node test/ink-preview.mjs   # writes ../ink-preview.svg
```

## Verification

Run from `frontend/`.

- `npx tsc --noEmit` — clean.
- `npm run build` — `Compiled successfully`.
- `npm test` — 54 assertions: 16/16 worklet, 9/9 stream, 29/29 ink geometry.
- Served the production build and confirmed the six marks, the trail, the
  `--depth` parallax values, the `translate: calc(var(--ax) …)` rule, the
  `data-awake` rule and the ENDLESS `@font-face` all reach the output.
- Visual check against the reference: matched on the joined pair at three
  separations, a row of four, and a diagonal run.
- A grep for `MetaballField`, `metaball-field`, `DotAmbience`, `dot-ambience` and
  `--blob-` across `app components lib test` returns nothing.

The ink geometry is tested, not asserted. `test/ink-render.test.mjs` transpiles
`lib/ink-render.ts` in process — the pattern the existing stream test already
uses — and runs it against a stub 2D context that records path calls. The checks
that carry real weight:

| Check | Why it matters |
|---|---|
| the handle lies along the circle's tangent (both ends, to 1e-9) | the property pass one got wrong, and the whole reason this reads as liquid |
| contact points sit exactly on their circles | the membrane meets the dots rather than near them |
| the membrane pinches inward at its waist, and stays outside the axis | concave, and not collapsed through itself |
| every subpath winds the same way | opposite windings cancel under nonzero fill and would hole every join |
| a pair beyond reach / a circle inside another / a zero radius is not joined | the degenerate cases upstream guards, still guarded |
| a saturated 2x2 block is plugged, one with an unreachable side is not | the pinhole fix, and its gate |
| different shades do not fuse | keeps the cat's rim an outline |
| dry shade draws cells and no membranes | the unlit lattice stays a lattice |
| a lone diagonal pair fuses, a solid block fuses orthogonally only | diagonals flow without packing corners |
| the faintest ink cannot reach its neighbour | ink joins as it grows rather than being wired together |

One test failed on first run and was **the test's** fault, not the code's: it
assumed a single faint corner would open a 2x2 square. It does not — `minRadius`
is deliberately large enough that a faint cell still reaches a *saturated*
neighbour, and only faint-to-faint fails to span the pitch. The test now asserts
that property explicitly and uses a faint *pair* for the negative case.

## Result

New: `frontend/lib/ink-render.ts`, `frontend/components/ascii-ambience.tsx`,
`frontend/test/ink-render.test.mjs`, `frontend/test/ink-preview.mjs`.
Deleted: `frontend/components/metaball-field.tsx`,
`frontend/components/dot-ambience.tsx`.
Modified: `dot-visualizer.tsx`, `pet.tsx`, `lib/pet-scene.ts`, `app/page.tsx`,
`app/globals.css`, `package.json`, `README.md`.

## Known gap: the ENDLESS font file

ENDLESS is not redistributable from here and I could not fetch it. Behance
returns 403 to a non-browser client; the titanui mirror's download endpoint
returns an empty body without a browser session. Both were tried.

So the wiring is in and the file is not. `app/globals.css` declares

```css
@font-face {
  font-family: "ENDLESS";
  src: url("/fonts/endless.woff2") format("woff2");
  font-display: swap;
}
```

A missing file at that path fails silently and the ASCII layer falls back to the
monospace stack — it does not break the build or the page. To finish it:

1. Download the family from
   <https://www.behance.net/gallery/247864363/ENDLESS-Geometric-Sans-Serif-Free-Font>
2. Convert the OTF/TTF to woff2.
3. Save it as `frontend/public/fonts/endless.woff2`.

No code change is needed once the file is there.
