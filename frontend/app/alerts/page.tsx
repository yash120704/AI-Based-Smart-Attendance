"use client";

import { useEffect, useState } from "react";
import { ShieldAlert } from "lucide-react";
import { AttendanceRecord, getProxyAlerts } from "@/lib/api";
import { formatDateTime, formatPercent } from "@/lib/format";

export default function ProxyAlerts() {
  const [alerts, setAlerts] = useState<AttendanceRecord[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await getProxyAlerts();
        if (!active) return;
        setAlerts(response);
        setError("");
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load proxy alerts");
      }
    }

    load();
    const interval = window.setInterval(load, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal text-zinc-950">Proxy Alerts</h1>
          <p className="text-sm text-zinc-500">{alerts.length} total alerts</p>
        </div>
        <div className="inline-flex w-fit items-center gap-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-700">
          <ShieldAlert size={16} aria-hidden="true" />
          Fraud Monitor
        </div>
      </div>

      {error ? <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}

      <section className="rounded-lg border border-zinc-200 bg-white">
        <div className="table-scroll">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-zinc-50 text-xs uppercase text-zinc-500">
              <tr>
                <th className="px-4 py-3 font-semibold">Person</th>
                <th className="px-4 py-3 font-semibold">Timestamp</th>
                <th className="px-4 py-3 font-semibold">Alert</th>
                <th className="px-4 py-3 font-semibold">Face</th>
                <th className="px-4 py-3 font-semibold">Attempts</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-rose-100">
              {alerts.map((alert, index) => (
                <tr key={`${alert.id || index}-${alert.timestamp}`} className="bg-rose-50/70 hover:bg-rose-100/70">
                  <td className="px-4 py-3 font-medium text-zinc-950">{alert.person_name}</td>
                  <td className="px-4 py-3 text-zinc-700">{formatDateTime(alert.timestamp)}</td>
                  <td className="max-w-xl px-4 py-3 text-rose-800">{alert.alert_message || "-"}</td>
                  <td className="px-4 py-3 text-zinc-700">{formatPercent(alert.face_confidence)}</td>
                  <td className="px-4 py-3 text-zinc-700">{alert.attempts ?? 0}</td>
                </tr>
              ))}
              {alerts.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-sm text-zinc-500">
                    No proxy alerts found
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
