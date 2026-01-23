import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/layout/Navbar";
import { InterventionToast } from "@/components/ui/intervention-toast";
import { cn } from "@/lib/utils";
import { AuthProvider } from "@/lib/auth-context";
import { TelemetryProvider } from "@/components/providers/telemetry-provider";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });

export const metadata: Metadata = {
  title: "NerdLearn - Master Complex Topics",
  description: "Adaptive learning platform with gamified mechanics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={cn(inter.variable, outfit.variable, "font-sans min-h-screen bg-background text-foreground antialiased")} suppressHydrationWarning>
        <AuthProvider>
          <TelemetryProvider>
            <div className="relative flex min-h-screen flex-col">
              <Navbar />
              <main className="flex-1 pt-16">
                {children}
              </main>
              <InterventionToast />
            </div>
          </TelemetryProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
