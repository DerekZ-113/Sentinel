/**
 * Shared slide-over shell: backdrop + right-side dialog panel with an
 * Escape-to-close listener. Header and content live with the caller.
 */

import { useEffect, type ReactNode } from "react";

interface DrawerShellProps {
  open: boolean;
  onClose: () => void;
  ariaLabel: string;
  widthClassName?: string;
  children: ReactNode;
}

export default function DrawerShell({
  open,
  onClose,
  ariaLabel,
  widthClassName = "w-full sm:w-[420px]",
  children,
}: DrawerShellProps) {
  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <aside
        className={`fixed inset-y-0 right-0 z-50 bg-gray-900 border-l border-gray-800 overflow-y-auto ${widthClassName}`}
        role="dialog"
        aria-label={ariaLabel}
      >
        {children}
      </aside>
    </>
  );
}
