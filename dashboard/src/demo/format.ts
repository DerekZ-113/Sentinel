/**
 * Time formatting helpers for demo-mode components.
 */

export function relativeTime(nowMs: number, iso: string): string {
  const secs = Math.max(0, Math.floor((nowMs - Date.parse(iso)) / 1000));
  if (secs < 2) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
}

export function clockTime(nowMs: number): string {
  return new Date(nowMs).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function minuteLabel(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}
