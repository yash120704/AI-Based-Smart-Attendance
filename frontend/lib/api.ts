const API_BASE = process.env.NEXT_PUBLIC_API_URL;

export type AttendanceRecord = {
  id?: number;
  person_name: string;
  timestamp: string;
  face_confidence?: number | null;
  behavior_confidence?: number | null;
  is_proxy?: boolean | number;
  alert_message?: string | null;
  attempts?: number | null;
  status?: string | null;
  blink_detected?: boolean | number;
};

export type PersonRecord = {
  name: string;
  registered_at: string;
  total_attendances: number;
  blocked: boolean | number;
  total_records?: number;
  proxy_count?: number | null;
};

export type Stats = {
  total_attended_today: number;
  fraud_alerts: number;
  active_persons: number;
  fps: number;
  model_loaded: boolean;
};

export type VerifyFrameResponse = {
  stage: "FACE" | "BLINK" | "BEHAVIOR" | "SUCCESS" | "RETRY" | "BLOCKED";
  person?: string | null;
  person_name?: string | null;
  face_confidence: number;
  behavior_confidence: number;
  elapsed_seconds: number;
  attempt: number;
  message: string;
  attendance_marked: boolean;
  decision: string;
};

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE) {
    throw new Error("NEXT_PUBLIC_API_URL is not configured");
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function recordsFromResponse<T>(response: { records?: T[] } | T[]): T[] {
  return Array.isArray(response) ? response : response.records || [];
}

export async function getTodayAttendance() {
  const response = await apiRequest<{ records: AttendanceRecord[] } | AttendanceRecord[]>("/api/attendance/today");
  return recordsFromResponse<AttendanceRecord>(response);
}

export async function getHistory(start: string, end: string) {
  const response = await apiRequest<{ records: AttendanceRecord[] } | AttendanceRecord[]>(
    `/api/attendance/history?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
  );
  return recordsFromResponse<AttendanceRecord>(response);
}

export async function getPersons() {
  const response = await apiRequest<{ persons: PersonRecord[] } | PersonRecord[]>("/api/persons");
  return Array.isArray(response) ? response : response.persons || [];
}

export async function blockPerson(name: string) {
  return apiRequest(`/api/persons/${encodeURIComponent(name)}/block`, { method: "POST" });
}

export async function unblockPerson(name: string) {
  return apiRequest(`/api/persons/${encodeURIComponent(name)}/unblock`, { method: "POST" });
}

export async function reenablePerson(name: string) {
  return apiRequest(`/api/persons/${encodeURIComponent(name)}/reenable`, { method: "POST" });
}

export async function getProxyAlerts() {
  const response = await apiRequest<{ alerts: AttendanceRecord[] } | AttendanceRecord[]>("/api/proxy-alerts");
  return Array.isArray(response) ? response : response.alerts || [];
}

export async function getStats() {
  return apiRequest<Stats>("/api/stats");
}

export async function verifyFrame(frame: string, sessionId: string) {
  return apiRequest<VerifyFrameResponse>("/api/verify-frame", {
    method: "POST",
    body: JSON.stringify({ frame, session_id: sessionId }),
  });
}

export async function trainModels() {
  return apiRequest("/api/train", { method: "POST" });
}
