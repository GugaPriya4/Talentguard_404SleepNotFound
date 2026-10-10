import clsx from "clsx";

const ACCENTS = {
  slate: "text-slate-900",
  red: "text-red-600",
  amber: "text-amber-600",
  emerald: "text-emerald-600",
  indigo: "text-indigo-600",
} as const;

export default function KpiCard({
  label,
  value,
  sub,
  accent = "slate",
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: keyof typeof ACCENTS;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:shadow-md">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className={clsx("mt-2 text-3xl font-bold tabular-nums", ACCENTS[accent])}>
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-slate-400">{sub}</p>}
    </div>
  );
}