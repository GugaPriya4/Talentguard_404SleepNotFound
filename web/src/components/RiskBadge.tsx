import clsx from "clsx";
import type { RiskTier } from "@/lib/types";

const STYLES: Record<string, string> = {
  Critical: "bg-red-50 text-red-700 ring-red-600/20",
  High: "bg-orange-50 text-orange-700 ring-orange-600/20",
  Elevated: "bg-amber-50 text-amber-700 ring-amber-600/20",
  Moderate: "bg-sky-50 text-sky-700 ring-sky-600/20",
  Low: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
};

export default function RiskBadge({ tier }: { tier: RiskTier }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ring-inset",
        STYLES[tier] ?? STYLES.Low
      )}
    >
      {tier}
    </span>
  );
}