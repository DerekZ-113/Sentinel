/**
 * Literal hex values for Recharts/SVG presentation attributes — SVG
 * fill/stroke attrs can't resolve CSS var(), so charts mirror the
 * @theme tokens in src/index.css here. chartTheme.test.ts guards the
 * two against drifting apart.
 */

export const CHART = {
  canvas: "#0B0E11",
  panel: "#10141A",
  inset: "#0C1015",
  hairline: "#1F262E",
  line: "#161C23",
  hairlineStrong: "#2A323B",
  ink: "#DEE5EC",
  inkMid: "#94A0AB",
  inkLow: "#5A6672",
  inkMicro: "#7A8791",
  inkData: "#B9C4CE",
  ok: "#3FB950",
  warn: "#D9A040",
  crit: "#E5534B",
  accent: "#56B4D3",
} as const;

export const CHART_FONT = '"IBM Plex Mono", ui-monospace, monospace';

export const CONF_COLORS = {
  high: CHART.ok,
  medium: CHART.warn,
  low: CHART.crit,
} as const;

export const AXIS_TICK = {
  fill: CHART.inkMicro,
  fontSize: 10,
  fontFamily: CHART_FONT,
} as const;

export const TOOLTIP_STYLE = {
  backgroundColor: CHART.inset,
  border: `1px solid ${CHART.hairlineStrong}`,
  borderRadius: 2,
  color: CHART.ink,
  fontSize: 11,
  fontFamily: CHART_FONT,
} as const;

export const LEGEND_STYLE = {
  fontSize: 11,
  color: CHART.inkMid,
  fontFamily: CHART_FONT,
} as const;
