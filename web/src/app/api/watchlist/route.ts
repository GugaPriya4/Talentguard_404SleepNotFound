import { NextResponse } from "next/server";
import { watchlist, riskRegister } from "@/lib/data";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const dept = searchParams.get("department");
  const tier = searchParams.get("tier");
  const limit = Number(searchParams.get("limit") ?? 50);
  const all = searchParams.get("all") === "true";

  let rows = all ? riskRegister : watchlist;
  if (dept) rows = rows.filter((r) => r.Department === dept);
  if (tier) rows = rows.filter((r) => r.RiskTier === tier);

  return NextResponse.json({ count: rows.length, results: rows.slice(0, limit) });
}