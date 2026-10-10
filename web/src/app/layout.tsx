import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TalentGuard — Attrition Intelligence",
  description: "AI-powered workforce retention analytics",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 antialiased">{children}</body>
    </html>
  );
}