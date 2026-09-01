/**
 * Ops-display value transition: the number snaps to the new value
 * synchronously (tests can assert textContent immediately) and a brief
 * background tint (.value-flash, 500ms) marks the change. The span is
 * keyed by a change counter so the CSS animation restarts on every
 * change; prefers-reduced-motion disables it in CSS.
 */

import { useEffect, useRef, useState } from "react";

interface ValueFlashProps {
  value: number;
  format?: (n: number) => string;
}

export default function ValueFlash({
  value,
  format = (n: number) => Math.round(n).toLocaleString(),
}: ValueFlashProps) {
  const [flashKey, setFlashKey] = useState(0);
  const prevRef = useRef(value);

  useEffect(() => {
    if (prevRef.current !== value) {
      prevRef.current = value;
      setFlashKey((k) => k + 1);
    }
  }, [value]);

  return (
    <span
      key={flashKey}
      className={`tabular-nums ${flashKey > 0 ? "value-flash" : ""}`}
    >
      {format(value)}
    </span>
  );
}
