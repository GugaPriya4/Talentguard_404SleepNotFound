"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import type { TierStat } from "@/lib/types";

const ORDER = ["Critical", "High", "Elevated", "Moderate", "Low"];
const COLORS: Record<string, string> = {
  Critical: "#ef4444",
  High: "#f97316",
  Elevated: "#f59e0b",
  Moderate: "#0ea5e9",
  Low: "#10b981",
};

export default function TierChart({ tiers }: { tiers: TierStat[] }) {
  const data = ORDER.map((t) => tiers.find((x) => x.tier === t)).filter(
    Boolean
  ) as TierStat[];

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-800">
        Risk tier validation
      </h3>
      <p className="mb-4 text-xs text-slate-400">
        Actual historical attrition rate per predicted tier
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="tier" tick={{ fontSize: 12, fill: "#475569" }} />
          <YAxis
            tick={{ fontSize: 11, fill: "#64748b" }}
            unit="%"
            domain={[0, 100]}
          />
          <Tooltip
            formatter={(v: number) => `${v}%`}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Bar dataKey="attritionRate" radius={[6, 6, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.tier} fill={COLORS[d.tier]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-3 grid grid-cols-5 gap-2 text-center">
        {data.map((d) => (
          <div key={d.tier}>
            <p className="text-sm font-semibold text-slate-800">{d.employees}</p>
            <p className="text-[10px] text-slate-400">employees</p>
          </div>
        ))}
      </div>
    </div>
  );
}