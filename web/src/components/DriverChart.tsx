"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { Driver } from "@/lib/types";

export default function DriverChart({ drivers }: { drivers: Driver[] }) {
  const data = drivers
    .slice(0, 12)
    .map((d) => ({ name: d.Label, value: Number(d.MeanAbsSHAP) }))
    .reverse();

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-800">
        Global attrition drivers
      </h3>
      <p className="mb-4 text-xs text-slate-400">
        Mean absolute SHAP value across all employees
      </p>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ left: 150, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
          <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} />
          <YAxis
            type="category"
            dataKey="name"
            width={145}
            tick={{ fontSize: 11, fill: "#475569" }}
          />
          <Tooltip
            formatter={(v: number) => v.toFixed(4)}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Bar dataKey="value" fill="#4a9eed" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}