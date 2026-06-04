"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, Gauge, Users } from "lucide-react";
import { AttendanceRecord, Stats, getStats, getTodayAttendance } from "@/lib/api";
import { formatDateTime, formatPercent, statusClass } from "@/lib/format";

function StatusBadge({ status }: { status?: string | null }) {
  return (
    <span className={`inline-flex min-w-20 items-center justify-center rounded-md border px-2 py-1 text-xs font-semibold ${statusClass(status)}`}>
      {status || "PENDING"}
    </span>
  );
}

function Metric({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  icon: typeof Activity;
}) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-zinc-500">{label}</p>
        <Icon className="text-zinc-400" size={18} aria-hidden="true" />
      </div>
      <p className="mt-3 text-3xl font-semibold tracking-normal text-zinc-950">{value}</p>
    </div>
  );
}

export default function LiveDashboard() {
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const [today, liveStats] = await Promise.all([getTodayAttendance(), getStats()]);
        if (!active) return;
        setRecords(today);
        setStats(liveStats);
        setError("");
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load dashboard data");
      }
    }

    load();
    const interval = window.setInterval(load, 3000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const recentRecords = useMemo(() => records.slice(0, 12), [records]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal text-zinc-950">Live Dashboard</h1>
          <p className="text-sm text-zinc-500">{new Date().toLocaleDateString()}</p>
        </div>
        <div className="inline-flex w-fit items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-600">
          <Gauge size={16} aria-hidden="true" />
          FPS {stats?.fps?.toFixed(1) ?? "0.0"}
        </div>
      </div>

      {error ? <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}

      <section className="grid gap-4 md:grid-cols-3">
        <Metric label="Total Attended Today" value={stats?.total_attended_today ?? 0} icon={Activity} />
        <Metric label="Fraud Alerts" value={stats?.fraud_alerts ?? 0} icon={AlertTriangle} />
        <Metric label="Active Persons" value={stats?.active_persons ?? 0} icon={Users} />
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white">
        <div className="border-b border-zinc-200 px-4 py-3">
          <h2 className="text-base font-semibold tracking-normal text-zinc-950">Live Attendance Feed</h2>
        </div>
        <div className="table-scroll">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-zinc-50 text-xs uppercase text-zinc-500">
              <tr>
                <th className="px-4 py-3 font-semibold">Person</th>
                <th className="px-4 py-3 font-semibold">Timestamp</th>
                <th className="px-4 py-3 font-semibold">Face</th>
                <th className="px-4 py-3 font-semibold">Behavior</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Message</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {recentRecords.map((record, index) => (
                <tr key={`${record.id || index}-${record.timestamp}`} className="hover:bg-zinc-50">
                  <td className="px-4 py-3 font-medium text-zinc-950">{record.person_name}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatDateTime(record.timestamp)}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatPercent(record.face_confidence)}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatPercent(record.behavior_confidence)}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={record.status} />
                  </td>
                  <td className="max-w-sm px-4 py-3 text-zinc-600">{record.alert_message || "-"}</td>
                </tr>
              ))}
              {recentRecords.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-sm text-zinc-500">
                    No attendance records today
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white">
        <div className="border-b border-zinc-200 px-4 py-3">
          <h2 className="text-base font-semibold tracking-normal text-zinc-950">Recent Activity</h2>
        </div>
        <div className="divide-y divide-zinc-100">
          {recentRecords.slice(0, 6).map((record, index) => (
            <div key={`activity-${record.id || index}`} className="flex flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="font-medium text-zinc-950">{record.person_name}</p>
                <p className="text-sm text-zinc-500">{formatDateTime(record.timestamp)}</p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm text-zinc-500">Face {formatPercent(record.face_confidence)}</span>
                <span className="text-sm text-zinc-500">Behavior {formatPercent(record.behavior_confidence)}</span>
                <StatusBadge status={record.status} />
              </div>
            </div>
          ))}
          {recentRecords.length === 0 ? <div className="px-4 py-8 text-center text-sm text-zinc-500">No recent activity</div> : null}
        </div>
      </section>
    </div>
  );
}
