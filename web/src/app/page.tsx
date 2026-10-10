import KpiCard from "@/components/KpiCard";
import InsightCard from "@/components/InsightCard";
import DriverChart from "@/components/DriverChart";
import TierChart from "@/components/TierChart";
import RiskTable from "@/components/RiskTable";
import {
  kpis,
  riskRegister,
  drivers,
  insights,
  recommendations,
} from "@/lib/data";

export default function Dashboard() {
  const m = kpis.modelMetrics;

  return (
    <main className="mx-auto max-w-7xl space-y-8 px-6 py-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">
            TalentGuard
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Employee Attrition Intelligence · {kpis.modelName} v
            {kpis.modelVersion}
          </p>
        </div>
        <div className="flex gap-4 text-right text-xs text-slate-500">
          <div>
            <p className="font-bold text-slate-800">{m.prAuc}</p>
            <p>PR-AUC</p>
          </div>
          <div>
            <p className="font-bold text-slate-800">{m.rocAuc}</p>
            <p>ROC-AUC</p>
          </div>
          <div>
            <p className="font-bold text-emerald-600">
              {(m.precisionAt50 * 100).toFixed(0)}%
            </p>
            <p>Precision@50</p>
          </div>
          <div>
            <p className="font-bold text-indigo-600">{m.liftAt50}x</p>
            <p>Lift@50</p>
          </div>
        </div>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          label="Total employees"
          value={kpis.totalEmployees.toLocaleString()}
        />
        <KpiCard
          label="Historical attrition"
          value={`${kpis.historicalAttritionRate}%`}
          sub={`${kpis.historicalAttritionCount} departures`}
          accent="amber"
        />
        <KpiCard
          label="High / critical risk"
          value={kpis.highRiskCount.toLocaleString()}
          sub="Requires HR review"
          accent="red"
        />
        <KpiCard
          label="Overtime vs non-overtime"
          value={`${kpis.overtimeAttritionRate}% / ${kpis.nonOvertimeAttritionRate}%`}
          sub="Attrition rate comparison"
          accent="red"
        />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold text-slate-800">
          Key findings
        </h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {insights.map((i) => (
            <InsightCard key={i.id} insight={i} />
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <TierChart tiers={kpis.riskTierStats} />
        <DriverChart drivers={drivers} />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-slate-800">
          Employee risk register
        </h2>
        <p className="text-xs text-slate-500">
          Click any row to view SHAP explanations and agent recommendations.
          Recommendations are generated for the top {recommendations.length > 0 ? 50 : 0} watchlist employees.
        </p>
        <RiskTable rows={riskRegister} recommendations={recommendations} />
      </section>

      <footer className="border-t border-slate-200 pt-6 text-xs text-slate-400">
        Generated {new Date(kpis.generatedAt).toLocaleString("en-GB")} · Model
        outputs are advisory and require human review. Not for use in
        termination, promotion or compensation decisions.
      </footer>
    </main>
  );
}