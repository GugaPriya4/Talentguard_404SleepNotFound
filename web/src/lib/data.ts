import kpisJson from "../../public/data/kpis.json";
import registerJson from "../../public/data/risk-register.json";
import watchlistJson from "../../public/data/watchlist.json";
import driversJson from "../../public/data/drivers.json";
import recsJson from "../../public/data/recommendations.json";
import insightsJson from "../../public/data/insights.json";

import type {
  Kpis,
  Employee,
  Driver,
  Recommendation,
  Insight,
} from "./types";

export const kpis = kpisJson as unknown as Kpis;
export const riskRegister = registerJson as unknown as Employee[];
export const watchlist = watchlistJson as unknown as Employee[];
export const drivers = driversJson as unknown as Driver[];
export const recommendations = recsJson as unknown as Recommendation[];
export const insights = insightsJson as unknown as Insight[];

export function getEmployee(id: number) {
  const employee = riskRegister.find((e) => e.employeeId === id);
  if (!employee) return null;
  return {
    employee,
    recommendations: recommendations.filter((r) => r.employeeId === id),
  };
}