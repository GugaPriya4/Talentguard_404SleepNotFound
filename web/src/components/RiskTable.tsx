"use client";

import { useMemo, useState } from "react";
import RiskBadge from "./RiskBadge";
import type { Employee, Recommendation } from "@/lib/types";


export default function RiskTable({
  rows,
  recommendations,
}: {
  rows: Employee[];
  recommendations: Recommendation[];
}) {
  const [dept, setDept] = useState("All");
  const [tier, setTier] = useState("All");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Employee | null>(null);

  const departments = useMemo(
    () => ["All", ...Array.from(new Set(rows.map((r) => r.Department)))],
    [rows]
  );
  const tiers = useMemo(
    () => ["All", ...Array.from(new Set(rows.map((r) => r.RiskTier)))],
    [rows]
  );

  const filtered = rows.filter(
    (r) =>
      (dept === "All" || r.Department === dept) &&
      (tier === "All" || r.RiskTier === tier) &&
      (query === "" ||
        String(r.employeeId).includes(query) ||
        r.JobRole.toLowerCase().includes(query.toLowerCase()))
  );

  const actions = selected
    ? recommendations.filter((r) => r.employeeId === selected.employeeId)
    : [];

  return (
    <>
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap gap-3 border-b border-slate-200 p-4">
          <select
            value={dept}
            onChange={(e) => setDept(e.target.value)}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
          >
            {departments.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
          <select
            value={tier}
            onChange={(e) => setTier(e.target.value)}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
          >
            {tiers.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search employee ID or role..."
            className="min-w-[200px] flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"
          />
          <span className="self-center text-sm text-slate-500">
            {filtered.length} shown
          </span>
        </div>

        <div className="max-h-[560px] overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-slate-50 text-left text-[11px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3">ID</th>
                <th className="px-4 py-3">Role</th>
                <th className="px-4 py-3">Dept</th>
                <th className="px-4 py-3">Lvl</th>
                <th className="px-4 py-3">OT</th>
                <th className="px-4 py-3">Income</th>
                <th className="px-4 py-3">Stock</th>
                <th className="px-4 py-3">Risk</th>
                <th className="px-4 py-3">Tier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filtered.map((r) => (
                <tr
                  key={r.employeeId}
                  onClick={() => setSelected(r)}
                  className="cursor-pointer hover:bg-sky-50"
                >
                  <td className="px-4 py-3 font-mono text-xs">{r.employeeId}</td>
                  <td className="px-4 py-3">{r.JobRole}</td>
                  <td className="px-4 py-3 text-slate-500">{r.Department}</td>
                  <td className="px-4 py-3">{r.JobLevel}</td>
                  <td className="px-4 py-3">
                    <span
                      className={
                        r.OverTime === "Yes"
                          ? "font-semibold text-red-600"
                          : "text-slate-400"
                      }
                    >
                      {r.OverTime}
                    </span>
                  </td>
                  <td className="px-4 py-3 tabular-nums">
                    {r.MonthlyIncome.toLocaleString()}
                  </td>
                  <td className="px-4 py-3">{r.StockOptionLevel}</td>
                  <td className="px-4 py-3 font-bold tabular-nums">
                    {r.RiskScore}%
                  </td>
                  <td className="px-4 py-3">
                    <RiskBadge tier={r.RiskTier} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-h-[85vh] w-full max-w-2xl overflow-auto rounded-2xl bg-white p-6 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  Employee {selected.employeeId}
                </p>
                <h3 className="text-xl font-bold text-slate-900">
                  {selected.JobRole}
                </h3>
                <p className="text-sm text-slate-500">
                  {selected.Department} · Level {selected.JobLevel} · Age{" "}
                  {selected.Age}
                </p>
              </div>
              <div className="text-right">
                <p className="text-3xl font-bold text-red-600">
                  {selected.RiskScore}%
                </p>
                <RiskBadge tier={selected.RiskTier} />
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["Income", selected.MonthlyIncome.toLocaleString()],
                ["Overtime", selected.OverTime],
                ["Stock options", selected.StockOptionLevel],
                ["Tenure", `${selected.YearsAtCompany} yrs`],
                ["Job satisfaction", `${selected.JobSatisfaction}/4`],
                ["Environment", `${selected.EnvironmentSatisfaction}/4`],
                ["Work-life balance", `${selected.WorkLifeBalance}/4`],
                ["Commute", selected.DistanceFromHome],
              ].map(([k, v]) => (
                <div key={k as string} className="rounded-lg bg-slate-50 p-3">
                  <p className="text-[10px] uppercase text-slate-400">{k}</p>
                  <p className="text-sm font-semibold text-slate-800">{v}</p>
                </div>
              ))}
            </div>

            <h4 className="mt-6 text-sm font-semibold text-slate-800">
              Top contributing factors (SHAP)
            </h4>
            <div className="mt-2 space-y-2">
              {[
                [selected.driver_1_label, selected.driver_1_shap],
                [selected.driver_2_label, selected.driver_2_shap],
                [selected.driver_3_label, selected.driver_3_shap],
              ].map(([label, shap], i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2"
                >
                  <span className="text-sm text-slate-700">{label}</span>
                  <span
                    className={
                      Number(shap) > 0
                        ? "text-xs font-semibold text-red-600"
                        : "text-xs font-semibold text-emerald-600"
                    }
                  >
                    {Number(shap) > 0 ? "increases" : "reduces"} risk (
                    {Number(shap).toFixed(3)})
                  </span>
                </div>
              ))}
            </div>

            {actions.length > 0 && (
              <>
                <h4 className="mt-6 text-sm font-semibold text-slate-800">
                  Agent recommendations
                </h4>
                <div className="mt-2 space-y-2">
                  {actions.map((a, i) => (
                    <div
                      key={i}
                      className="rounded-lg border border-slate-200 bg-slate-50 p-3"
                    >
                      <p className="text-xs font-semibold text-indigo-600">
                        {a.Agent}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        Trigger: {a.Trigger}
                      </p>
                      <p className="mt-1 text-sm text-slate-800">
                        {a.RecommendedAction}
                      </p>
                    </div>
                  ))}
                </div>
              </>
            )}

            <button
              onClick={() => setSelected(null)}
              className="mt-6 w-full rounded-lg bg-slate-900 py-2.5 text-sm font-medium text-white hover:bg-slate-700"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}