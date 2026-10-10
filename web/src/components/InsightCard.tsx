import clsx from "clsx";
import type { Insight } from "@/lib/types";

const BORDER: Record<string, string> = {
  critical: "border-l-red-500",
  high: "border-l-orange-500",
  medium: "border-l-amber-500",
  info: "border-l-sky-500",
};

export default function InsightCard({ insight }: { insight: Insight }) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-l-4 border-slate-200 bg-white p-5 shadow-sm",
        BORDER[insight.severity]
      )}
    >
      <p className="text-2xl font-bold text-slate-900">{insight.metric}</p>
      <p className="mt-1 text-sm font-semibold text-slate-700">
        {insight.title}
      </p>
      <p className="mt-2 text-xs leading-relaxed text-slate-500">
        {insight.detail}
      </p>
    </div>
  );
}