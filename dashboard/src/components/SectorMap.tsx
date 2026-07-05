/**
 * Stylized sector mini-map for the alert detail drawer. Pure SVG — no map
 * library. Positions come from the fixture lat/lon (Bay Area bounding box);
 * manual alerts without coordinates get a stable hash-based position.
 */

const LAT_MIN = 37.3;
const LAT_MAX = 37.8;
const LON_MIN = -122.5;
const LON_MAX = -122.0;

const COLS = 8;
const ROWS = 5;
const W = 240;
const H = 150;

/** Stable [0,1) hash for coordinate-less manual alerts. */
function hash01(seed: string, salt: number): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 10_000) / 10_000;
}

function clamp01(v: number): number {
  return Math.min(0.999, Math.max(0, v));
}

interface SectorMapProps {
  latitude?: number | null;
  longitude?: number | null;
  /** Fallback position seed (vehicle id) when coordinates are absent. */
  seed: string;
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export default function SectorMap({ latitude, longitude, seed }: SectorMapProps) {
  const animatePulse = !prefersReducedMotion();
  const nx =
    longitude !== null && longitude !== undefined
      ? clamp01((longitude - LON_MIN) / (LON_MAX - LON_MIN))
      : hash01(seed, 1);
  // Higher latitude renders toward the top
  const ny =
    latitude !== null && latitude !== undefined
      ? clamp01(1 - (latitude - LAT_MIN) / (LAT_MAX - LAT_MIN))
      : hash01(seed, 2);

  const x = nx * W;
  const y = ny * H;
  const col = Math.floor(nx * COLS);
  const row = Math.floor(ny * ROWS);
  const sector = `${String.fromCharCode(65 + row)}${col + 1}`;

  const vLines = Array.from({ length: COLS - 1 }, (_, i) => ((i + 1) * W) / COLS);
  const hLines = Array.from({ length: ROWS - 1 }, (_, i) => ((i + 1) * H) / ROWS);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded-lg border border-gray-700/50 bg-gray-900"
        role="img"
        aria-label={`Vehicle location, sector ${sector}`}
      >
        {/* Sector grid */}
        {vLines.map((gx) => (
          <line key={`v${gx}`} x1={gx} y1={0} x2={gx} y2={H} stroke="#1f2937" strokeWidth={1} />
        ))}
        {hLines.map((gy) => (
          <line key={`h${gy}`} x1={0} y1={gy} x2={W} y2={gy} stroke="#1f2937" strokeWidth={1} />
        ))}

        {/* Stylized streets */}
        <polyline
          points={`0,${H * 0.72} ${W * 0.3},${H * 0.66} ${W * 0.55},${H * 0.7} ${W},${H * 0.6}`}
          fill="none"
          stroke="#374151"
          strokeWidth={3}
          strokeLinecap="round"
        />
        <polyline
          points={`0,${H * 0.3} ${W * 0.45},${H * 0.34} ${W},${H * 0.26}`}
          fill="none"
          stroke="#374151"
          strokeWidth={2}
          strokeLinecap="round"
        />
        <polyline
          points={`${W * 0.25},0 ${W * 0.28},${H * 0.5} ${W * 0.22},${H}`}
          fill="none"
          stroke="#374151"
          strokeWidth={2}
          strokeLinecap="round"
        />
        <polyline
          points={`${W * 0.7},0 ${W * 0.66},${H * 0.55} ${W * 0.74},${H}`}
          fill="none"
          stroke="#374151"
          strokeWidth={2}
          strokeLinecap="round"
        />
        {/* Diagonal highway */}
        <line
          x1={0}
          y1={H * 0.05}
          x2={W}
          y2={H * 0.95}
          stroke="#4b5563"
          strokeWidth={2.5}
          strokeDasharray="8 5"
          opacity={0.5}
        />

        {/* Vehicle marker with pulse */}
        <circle cx={x} cy={y} r={4} fill="#3b82f6" opacity={0.35}>
          {animatePulse && (
            <>
              <animate attributeName="r" values="4;11;4" dur="2s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.35;0;0.35" dur="2s" repeatCount="indefinite" />
            </>
          )}
        </circle>
        <circle cx={x} cy={y} r={3.5} fill="#3b82f6" stroke="#bfdbfe" strokeWidth={1} />
      </svg>
      <span className="absolute top-1.5 right-2 text-[10px] font-mono text-gray-500">
        Sector {sector}
      </span>
    </div>
  );
}
