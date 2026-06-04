import type { AttendanceRecord } from "./api";

export function formatDateTime(value?: string | null) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export function isTruthy(value: boolean | number | undefined) {
  return value === true || value === 1;
}

export function statusClass(status?: string | null) {
  switch ((status || "").toUpperCase()) {
    case "SUCCESS":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "PROXY":
      return "border-rose-200 bg-rose-50 text-rose-700";
    case "BLOCKED":
      return "border-amber-200 bg-amber-50 text-amber-700";
    case "RETRY":
      return "border-sky-200 bg-sky-50 text-sky-700";
    default:
      return "border-zinc-200 bg-zinc-50 text-zinc-700";
  }
}

export function toCsv(records: AttendanceRecord[]) {
  const columns = [
    "person_name",
    "timestamp",
    "face_confidence",
    "behavior_confidence",
    "status",
    "blink_detected",
    "attempts",
    "alert_message",
  ];
  const escape = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [columns.join(","), ...records.map((record) => columns.map((column) => escape(record[column as keyof AttendanceRecord])).join(","))].join("\n");
}
