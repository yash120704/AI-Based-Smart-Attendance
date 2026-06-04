import type { Metadata } from "next";
import Link from "next/link";
import { Activity, CalendarDays, ScanFace, ShieldAlert, Users } from "lucide-react";
import "./globals.css";

const navigation = [
  { href: "/", label: "Live", icon: Activity },
  { href: "/attendance", label: "History", icon: CalendarDays },
  { href: "/persons", label: "Persons", icon: Users },
  { href: "/alerts", label: "Alerts", icon: ShieldAlert },
  { href: "/verify", label: "Verify", icon: ScanFace },
];

export const metadata: Metadata = {
  title: "Smart Attendance",
  description: "Smart Attendance cloud dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-zinc-100 text-zinc-950">
        <div className="min-h-screen">
          <header className="border-b border-zinc-200 bg-white">
            <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
              <Link href="/" className="flex items-center gap-3 text-lg font-semibold tracking-normal text-zinc-950">
                <span className="flex h-9 w-9 items-center justify-center rounded-md bg-zinc-950 text-white">
                  <ScanFace size={19} aria-hidden="true" />
                </span>
                Smart Attendance
              </Link>
              <nav className="flex flex-wrap gap-2">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="inline-flex h-10 items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 text-sm font-medium text-zinc-700 transition hover:border-zinc-300 hover:bg-zinc-50"
                    >
                      <Icon size={16} aria-hidden="true" />
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
