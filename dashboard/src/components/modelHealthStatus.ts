/**
 * Status→style map shared by ModelHealthView and CompactModelHealthCard.
 * Lives outside the component files so fast refresh stays intact
 * (react-refresh/only-export-components).
 */

export const STATUS_CONFIG: Record<
  string,
  { color: string; bg: string; label: string }
> = {
  healthy: { color: "text-ok", bg: "bg-ok", label: "Healthy" },
  warning: { color: "text-warn", bg: "bg-warn", label: "Warning" },
  degraded: { color: "text-crit", bg: "bg-crit", label: "Degraded" },
};
