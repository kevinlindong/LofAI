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
// float: every lump reaches back to a fixed lip, so the ring grows outward
// from it instead of lifting off the play button like a smoke ring.
//
// nothing else is drawn. the ring is the whole visualiser, and every mark on
// the panel is part of its surface - so anything the field puts outside that
// surface reads as dirt on the lens rather than as information.

import { BlobSet, SURFACE } from "./dot-field"

export const SPOKES = 18

// where a spoke's crest sits between the inner lip and the outer edge of the
// lattice, at silence and at full scale. the whole lattice is the range: a
// silent spoke lights one ring and a spoke at full scale lights all eleven,
// so the panel has somewhere to go on a loud bar instead of being most of the
// way full before the music starts.
const REST_OUT = 0.04
const LOUD_OUT = 1 - REST_OUT

// a blob's field keeps climbing for a way past its own radius, because the
// neighbours either side are still adding to it out there. this is where the
// surface actually lands once eighteen of them overlap, measured in radii.
//
// this is not a detail of taste. the crest is the only thing on the panel the
// eye reads as the level, so it has to land where the caller asked. sized the
// obvious way - centre at the crest, radius back to the lip - the surface
// comes out a third of a panel further out than intended, which is why a ring
// built that way runs off the edge and stays there for anything above about
// half scale. dividing the overshoot out is what buys back the top half of
// the range.
const OVERSHOOT = 1.14

// the smallest blob that still fuses with the one next to it, as a fraction of
// the gap between spokes. a quiet spoke wants a blob far smaller than this,
// and at silence the whole ring does - so without a floor the annulus breaks
// into eighteen separate lumps exactly when it is meant to be a thin quiet
// line. the value is set where the base ring is still unbroken all the way
// round; it stops binding at about a third scale, which is well below where
// the crest is doing any work.
const FUSE = 0.6

const TAU = Math.PI * 2
const COS = Float32Array.from({ length: SPOKES }, (_, s) => Math.cos(s / SPOKES * TAU))
const SIN = Float32Array.from({ length: SPOKES }, (_, s) => Math.sin(s / SPOKES * TAU))

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

export function buildWave(blobs: BlobSet, amp: Float32Array, geo: WaveGeometry) {
  blobs.reset()
  // the lip the ring grows out of, a shade inside the first ring of dots so
  // there is never a dark gap between the wave and the middle of the panel
  const lip = geo.inner - geo.pitch * 0.8
  const minR = ((TAU * geo.inner) / SPOKES) * FUSE

  for (let s = 0; s < SPOKES; s++) {
    // the radius the surface is to reach on this spoke, and the blob that puts
    // it there: wide enough to fill from the lip out to the crest, then walked
    // back in by however far its own field overshoots
    const crest = geo.inner + geo.span * (REST_OUT + Math.max(0, Math.min(1, amp[s])) * LOUD_OUT)
    const r = Math.max(minR, (crest - lip) * 0.5)
    const d = crest - r * OVERSHOOT
    blobs.add(COS[s] * d, SIN[s] * d, r)
  }
}
