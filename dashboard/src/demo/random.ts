/**
 * Seedable randomness helpers for the replay engine.
 */

export type Rng = () => number;

/** Small fast seedable PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sample an exponentially distributed gap, clamped to [minMs, maxMs]. */
export function expGap(rng: Rng, meanMs: number, minMs: number, maxMs: number): number {
  // Inverse CDF; rng() ∈ [0,1) so 1-u ∈ (0,1] and log is finite
  const gap = -meanMs * Math.log(1 - rng());
  return Math.min(maxMs, Math.max(minMs, gap));
}

/** Fisher-Yates shuffle into a new array. */
export function shuffle<T>(items: readonly T[], rng: Rng): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
