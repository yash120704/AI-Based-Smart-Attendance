"use client";

import { useEffect, useMemo, useState } from "react";
import { Download, Filter } from "lucide-react";
import { AttendanceRecord, getHistory } from "@/lib/api";
import { formatDateTime, formatPercent, isTruthy, statusClass, toCsv } from "@/lib/format";

function dateValue(offsetDays = 0) {
  const date = new Date();
  date.setDate(date.getDate() + offsetDays);
  return date.toISOString().slice(0, 10);
}

function StatusBadge({ status }: { status?: string | null }) {
  return (
    <span className={`inline-flex min-w-20 items-center justify-center rounded-md border px-2 py-1 text-xs font-semibold ${statusClass(status)}`}>
      {status || "PENDING"}
    </span>
  );
}

export default function AttendanceHistory() {
  const [start, setStart] = useState(() => dateValue(-30));
  const [end, setEnd] = useState(() => dateValue(0));
  const [personFilter, setPersonFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;

    async function load() {
      setLoading(true);
      try {
        const history = await getHistory(start, end);
        if (!active) return;
        setRecords(history);
        setError("");
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load attendance history");
      } finally {
        if (active) setLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, [start, end]);

  const filteredRecords = useMemo(() => {
    return records.filter((record) => {
      const personMatches = personFilter ? record.person_name.toLowerCase().includes(personFilter.toLowerCase()) : true;
      const statusMatches = statusFilter ? (record.status || "").toUpperCase() === statusFilter : true;
      return personMatches && statusMatches;
    });
  }, [records, personFilter, statusFilter]);

  function exportCsv() {
    const csv = toCsv(filteredRecords);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `attendance-${start}-to-${end}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal text-zinc-950">Attendance History</h1>
          <p className="text-sm text-zinc-500">{filteredRecords.length} records</p>
        </div>
        <button type="button" onClick={exportCsv} className="inline-flex h-10 w-fit items-center gap-2 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white transition hover:bg-zinc-800">
          <Download size={16} aria-hidden="true" />
          Export CSV
        </button>
      </div>

      {error ? <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}

      <section className="rounded-lg border border-zinc-200 bg-white p-4">
        <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-zinc-800">
          <Filter size={16} aria-hidden="true" />
          Filters
        </div>
        <div className="grid gap-3 md:grid-cols-4">
          <label className="space-y-1 text-sm font-medium text-zinc-700">
            Start
            <input type="date" value={start} onChange={(event) => setStart(event.target.value)} className="h-10 w-full rounded-md border border-zinc-300 px-3 text-sm outline-none transition focus:border-zinc-500" />
          </label>
          <label className="space-y-1 text-sm font-medium text-zinc-700">
            End
            <input type="date" value={end} onChange={(event) => setEnd(event.target.value)} className="h-10 w-full rounded-md border border-zinc-300 px-3 text-sm outline-none transition focus:border-zinc-500" />
          </label>
          <label className="space-y-1 text-sm font-medium text-zinc-700">
            Person
            <input value={personFilter} onChange={(event) => setPersonFilter(event.target.value)} className="h-10 w-full rounded-md border border-zinc-300 px-3 text-sm outline-none transition focus:border-zinc-500" placeholder="Name" />
          </label>
          <label className="space-y-1 text-sm font-medium text-zinc-700">
            Status
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} className="h-10 w-full rounded-md border border-zinc-300 px-3 text-sm outline-none transition focus:border-zinc-500">
              <option value="">All</option>
              <option value="SUCCESS">SUCCESS</option>
              <option value="PROXY">PROXY</option>
              <option value="BLOCKED">BLOCKED</option>
              <option value="RETRY">RETRY</option>
            </select>
          </label>
        </div>
      </section>

      <section className="rounded-lg border border-zinc-200 bg-white">
        <div className="border-b border-zinc-200 px-4 py-3">
          <h2 className="text-base font-semibold tracking-normal text-zinc-950">{loading ? "Loading Records" : "Records"}</h2>
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
                <th className="px-4 py-3 font-semibold">Blink</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {filteredRecords.map((record, index) => (
                <tr key={`${record.id || index}-${record.timestamp}`} className="hover:bg-zinc-50">
                  <td className="px-4 py-3 font-medium text-zinc-950">{record.person_name}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatDateTime(record.timestamp)}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatPercent(record.face_confidence)}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatPercent(record.behavior_confidence)}</td>
                  <td className="px-4 py-3"><StatusBadge status={record.status} /></td>
                  <td className="px-4 py-3 text-zinc-600">{isTruthy(record.blink_detected) ? "Yes" : "No"}</td>
                </tr>
              ))}
              {filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-sm text-zinc-500">No records found</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
