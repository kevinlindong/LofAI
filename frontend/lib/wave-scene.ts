// the music, as a ring of metaballs.
//
// one blob per band, set on a circle, each pushed outward by how loud its band
// is. the blobs are large enough that neighbours always overlap, so what you
// see is never eighteen circles - it is one gooey annulus whose edge bulges
// where the music is loud and pinches where it is quiet. a band that spikes
// grows a pseudopod; the ring either side of it stays put and the surface
// stretches between them, which is exactly the shape a bar chart bent into a
// circle cannot make.
//
// the inner edge is pinned near the middle of the panel rather than left to
// float: a blob's radius is taken from how far its centre has travelled, so
// the ring grows outward from a fixed lip instead of lifting off the play
// button like a smoke ring.
//
// each band also carries a peak-hold droplet. while the wave is loud it is
// swallowed by the lump under it; when the wave drops the droplet is left
// hanging where the transient got to, and sinks slowly back into the surface.
// that detach-and-rejoin is free here and impossible with bars.

import { BlobSet, SURFACE } from "./dot-field"

export const SPOKES = 18

// where the ring sits between the inner lip and the outer edge, at silence and
// at full scale. the top of the range overshoots the lattice a little on
// purpose, so a genuinely loud bar runs off the edge of the panel instead of
// politely stopping at it.
const REST_OUT = 0.12
const LOUD_OUT = 0.55

// how much of the distance from the lip to the blob's centre the blob's own
// radius covers. a shade over one, so every lump reaches back to the lip and
// the wave is filled behind its crest rather than being a hoop.
//
// this is not a detail of taste. a hoop three rings thick has an outer edge, an
// inner edge and almost nothing in between, and since the two edges land on
// different rings at different angles, the ring of dots ends up alternating
// between edge and body all the way round - which reads as speckle, not as
// shading. filled, there is always a body for the crest to be the edge of.
const FATNESS = 1.05

const TAU = Math.PI * 2

export { SURFACE }

// what the lattice paints once it knows which of its dots the field has
// caught. the outer edge is the brightest mark and the inner edge a rank
// dimmer, so the ring reads as lit from outside rather than as a flat washer.
export const WAVE_BODY = 4
export const WAVE_CREST = 5
export const WAVE_LIP = 3
export const WAVE_GLOW = 2

// how far below the surface a dot still gets a breath of glow
export const WAVE_GLOW_AT = SURFACE * 0.45

export interface WaveGeometry {
  // radius of the innermost ring of dots, in pixels
  inner: number
  // how much radius the lattice covers beyond that
  span: number
  // the radial distance between rings
  pitch: number
}

export function buildWave(
  blobs: BlobSet,
  amp: Float32Array,
  peak: Float32Array,
  spin: number,
  geo: WaveGeometry,
) {
  blobs.reset()
  // the lip the ring grows out of, a shade inside the first ring of dots so
  // there is never a dark gap between the wave and the middle of the panel
  const lip = geo.inner - geo.pitch * 0.8

  for (let s = 0; s < SPOKES; s++) {
    const a = (s / SPOKES) * TAU + spin
    const cos = Math.cos(a)
    const sin = Math.sin(a)

    const d = geo.inner + geo.span * (REST_OUT + amp[s] * LOUD_OUT)
    blobs.add(cos * d, sin * d, (d - lip) * FATNESS)

    // the droplet the transient left behind, once the wave under it has fallen
    // far enough away to let it show
    const p = geo.inner + geo.span * (REST_OUT + peak[s] * LOUD_OUT)
    if (p > d + geo.pitch * 1.4) blobs.add(cos * p, sin * p, geo.pitch * 0.9)
  }
}
