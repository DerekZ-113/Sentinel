# Sentinel Dashboard — Ground Control Design Language

The dashboard follows a fixed visual language informed by ground-control
stations and avionics displays: dense, flat, monospace-first, with color
reserved for state. This document is the contract; the tokens live in
`src/index.css` (`@theme`) and are mirrored for SVG in
`src/components/chartTheme.ts`. Don't restyle casually — propose changes
here first.

## 1. Tokens

| Token | Hex | Utility | Usage |
|---|---|---|---|
| `--color-canvas` | `#0B0E11` | `bg-canvas` | page background |
| `--color-header` | `#0D1117` | `bg-header` | top bar |
| `--color-panel` | `#10141A` | `bg-panel` | cards, drawers, sticky table headers |
| `--color-inset` | `#0C1015` | `bg-inset` | wells inside panels (cells, tracks, form controls) |
| `--color-hairline` | `#1F262E` | `border-hairline` | panel borders |
| `--color-line` | `#161C23` | `border-line` | row dividers, inset-cell borders, chart grids |
| `--color-hairline-2` | `#2A323B` | `border-hairline-2` | interactive outlines (buttons, inputs, table header rule) |
| `--color-ink` | `#DEE5EC` | `text-ink` | primary text, KPI values |
| `--color-ink-mid` | `#94A0AB` | `text-ink-mid` | panel titles, secondary controls |
| `--color-ink-low` | `#5A6672` | `text-ink-low` | tertiary text, timestamps, subtitles |
| `--color-ink-micro` | `#7A8791` | `text-ink-micro` | micro-labels |
| `--color-ink-data` | `#B9C4CE` | `text-ink-data` | table/data text |
| `--color-ok` | `#3FB950` | `*-ok` | normal / suppress / healthy |
| `--color-warn` | `#D9A040` | `*-warn` | caution / degraded-adjacent / SIM annunciator |
| `--color-crit` | `#E5534B` | `*-crit` | warning / flag / errors |
| `--color-accent` | `#56B4D3` | `*-accent` | THE interactive accent (see §4) |
| `--color-tag-*` | see index.css | `*-tag-*` | notification-type chips only |

Fonts: `--font-mono` = IBM Plex Mono (the **default**, set on `body`);
`--font-sans` = IBM Plex Sans (opt-in via `font-sans`, prose only).
Self-hosted via `@fontsource` imports in `src/main.tsx` — no font CDNs.

## 2. Typography recipes

| Recipe | Classes |
|---|---|
| Micro-label | `text-[10px] uppercase tracking-[0.1em] text-ink-micro` |
| Panel title | `text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-mid` |
| KPI value | `text-[22px] font-medium tabular-nums` |
| Data text | `text-xs text-ink-data` (+ `tabular-nums` for numbers) |
| Wordmark | `text-xs font-semibold uppercase tracking-[0.18em] text-ink` |

**Rule: uppercase is always CSS (`uppercase` utility), never typed into
textContent.** This keeps tests, accessible names, and copy stable.
Numerals are always `tabular-nums`. Units and timezones are written out
(`%`, `s`, `Z`).

## 3. Shape & surface

- Radius: `rounded-xs` (2px) is the **only** radius. Status dots are
  bare squares (no radius class).
- Borders: 1px hairlines everywhere; pick the role (`hairline` /
  `line` / `hairline-2`), never a gray.
- Fills are flat and opaque. **Banned:** shadows, `backdrop-blur`,
  gradients, translucent panel backgrounds, `animate-ping`/pulse,
  emoji/dingbat glyphs (use `src/components/icons.tsx`).
- Documented exception: `DrawerShell`'s scrim stays `bg-black/50`
  (an overlay, not a surface).

## 4. Color rules

- Color encodes **state only**: `ok` normal, `warn` caution, `crit`
  warning. Model statuses map Healthy→ok, Warning→warn, Degraded→crit
  (`src/components/modelHealthStatus.ts`).
- `accent` (cyan) is the single interactive hue. Allowed sites:
  Simulate / Run Prediction buttons (accent **outline**), the MANUAL
  chip, the stream filter chip, the jump-to-latest pill (the one
  **filled** accent element), links, `focus:border-accent`. Everything
  else interactive is a neutral outline (`border-hairline-2
  text-ink-mid`) — e.g. the Details button, pagination.
- Notification-type chips: `text-tag-X border-tag-X/40 bg-tag-X/14`,
  `text-[9.5px] uppercase tracking-[0.05em] rounded-xs`. Tag hues never
  appear outside type chips.
- State boxes: `bg-{ok|warn|crit}/10 border-{...}/40` with matching text.

## 5. Motion

Steady indicators (LIVE dot does not blink). Motion only marks data
changing: `.value-flash` (500ms background tint on KPI change, via
`ValueFlash`) and `.alert-in` (700ms tint on new stream rows). Both are
disabled under `prefers-reduced-motion`. No slides, spins, or count-ups.

## 6. Charts (`src/components/chartTheme.ts`)

SVG presentation attributes can't resolve `var()`, so Recharts/SVG code
imports literal hexes from `chartTheme.ts`, which mirrors `@theme`.
`src/__tests__/chartTheme.test.ts` fails if the two drift. Use
`AXIS_TICK`, `TOOLTIP_STYLE`, `LEGEND_STYLE`, `CONF_COLORS`; series
colors are `CHART.ok`/`CHART.crit` (state semantics carry into charts).
Bar corner radius is 2; area fills at ≤0.08 opacity are allowed.

## 7. Component patterns

- `VerdictChip` (alertRowParts): FLAG=crit / SUPPRESS=ok bordered chip.
- `TypeBadge`: type chip + subtype demoted to secondary text + full
  `type/subtype` tooltip.
- `ConfidenceBar`: % over a 3px `bg-inset` track; fill by thresholds
  ≥90 ok / ≥70 warn / else crit.
- SIM annunciator (`StatusBar`): warn-bordered chip, **always visible**
  — synthetic-data honesty is a mode flag, not a dismissible banner.
- Form controls (`SimulatePanel`): selects `bg-inset border-hairline-2
  focus:border-accent`; ranges `accent-accent`; toggles use state boxes.

## 8. Checklist for new UI

- [ ] No new colors, no Tailwind palette classes (`gray-*`, `blue-*`…)
- [ ] No new radii; `rounded-xs` or square
- [ ] Mono by default; micro-labels for captions; uppercase via CSS only
- [ ] Color only for state; neutral outlines for secondary actions
- [ ] Icons from `icons.tsx`, never glyph characters
- [ ] Charts pull from `chartTheme.ts`
