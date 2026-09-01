/**
 * Inline stroke icons — the app never uses emoji/dingbat glyphs as UI
 * (DESIGN.md). All icons inherit color via currentColor and are decorative
 * (aria-hidden); meaning is carried by the surrounding element's text or
 * aria-label.
 */

interface IconProps {
  size?: number;
  className?: string;
}

const strokeProps = {
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "square" as const,
  fill: "none",
};

export function IconX({ size = 12, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M2 2l8 8M10 2l-8 8" {...strokeProps} />
    </svg>
  );
}

export function IconCross({ size = 10, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M2 2l8 8M10 2l-8 8" {...strokeProps} />
    </svg>
  );
}

export function IconCheck({ size = 12, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M2 6.5l3 3 5-7" {...strokeProps} />
    </svg>
  );
}

export function IconArrowUp({ size = 10, className }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path d="M6 10V2M2.5 5.5L6 2l3.5 3.5" {...strokeProps} />
    </svg>
  );
}
