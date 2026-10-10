export type RiskTier = "Critical" | "High" | "Elevated" | "Moderate" | "Low";

export interface Employee {
  employeeId: number;
  Age: number;
  Department: string;
  JobRole: string;
  JobLevel: number;
  OverTime: string;
  MonthlyIncome: number;
  StockOptionLevel: number;
  YearsAtCompany: number;
  YearsInCurrentRole: number;
  YearsWithCurrManager: number;
  JobSatisfaction: number;
  EnvironmentSatisfaction: number;
  WorkLifeBalance: number;
  JobInvolvement: number;
  BusinessTravel: string;
  DistanceFromHome: number;
  RiskScore: number;
  RiskTier: RiskTier;
  top_factors: string;
  driver_1_label: string;
  driver_1_shap: number;
  driver_2_label: string;
  driver_2_shap: number;
  driver_3_label: string;
  driver_3_shap: number;
}

export interface TierStat {
  tier: RiskTier;
  employees: number;
  leavers: number;
  attritionRate: number;
  avgRiskScore: number;
}

export interface Kpis {
  generatedAt: string;
  modelName: string;
  modelVersion: string;
  totalEmployees: number;
  historicalAttritionCount: number;
  historicalAttritionRate: number;
  overtimeAttritionRate: number;
  nonOvertimeAttritionRate: number;
  highRiskCount: number;
  riskTierStats: TierStat[];
  modelMetrics: {
    prAuc: number;
    rocAuc: number;
    precisionAt50: number;
    recallAt50: number;
    liftAt50: number;
    brier: number;
  };
  operatingPoint: Record<string, number>;
  departmentBreakdown: {
    department: string;
    employees: number;
    attritionRate: number;
    avgRiskScore: number;
    highRisk: number;
  }[];
  jobRoleBreakdown: {
    jobRole: string;
    employees: number;
    attritionRate: number;
    avgRiskScore: number;
    highRisk: number;
  }[];
}

export interface Driver {
  Feature: string;
  Label: string;
  MeanAbsSHAP: number;
}

export interface Recommendation {
  employeeId: number;
  RiskScore: number;
  RiskTier: RiskTier;
  Agent: string;
  Trigger: string;
  RecommendedAction: string;
}

export interface Insight {
  id: string;
  severity: "critical" | "high" | "medium" | "info";
  metric: string;
  title: string;
  detail: string;
}