/**
 * Status→style map shared by ModelHealthView and CompactModelHealthCard.
 * Lives outside the component files so fast refresh stays intact
 * (react-refresh/only-export-components).
 */

export const STATUS_CONFIG: Record<
  string,
  { color: string; bg: string; label: string }
> = {
  healthy: { color: "text-emerald-400", bg: "bg-emerald-400", label: "Healthy" },
  warning: { color: "text-yellow-400", bg: "bg-yellow-400", label: "Warning" },
  degraded: { color: "text-red-400", bg: "bg-red-400", label: "Degraded" },
};
