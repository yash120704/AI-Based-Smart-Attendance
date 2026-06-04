"use client";

import { useEffect, useState } from "react";
import { LockKeyhole, RefreshCw, RotateCcw, UnlockKeyhole, Users } from "lucide-react";
import { PersonRecord, blockPerson, getPersons, reenablePerson, trainModels, unblockPerson } from "@/lib/api";
import { formatDateTime, isTruthy } from "@/lib/format";

function PersonStatus({ blocked }: { blocked: boolean | number }) {
  const isBlocked = isTruthy(blocked);
  return (
    <span className={`inline-flex min-w-20 items-center justify-center rounded-md border px-2 py-1 text-xs font-semibold ${isBlocked ? "border-amber-200 bg-amber-50 text-amber-700" : "border-emerald-200 bg-emerald-50 text-emerald-700"}`}>
      {isBlocked ? "BLOCKED" : "ACTIVE"}
    </span>
  );
}

export default function PersonManagement() {
  const [persons, setPersons] = useState<PersonRecord[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busyName, setBusyName] = useState("");

  async function loadPersons() {
    try {
      const response = await getPersons();
      setPersons(response);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load persons");
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadPersons();
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  async function runPersonAction(name: string, action: () => Promise<unknown>) {
    setBusyName(name);
    try {
      await action();
      setNotice("");
      await loadPersons();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setBusyName("");
    }
  }

  async function runTraining() {
    setBusyName("__train__");
    try {
      await trainModels();
      setNotice("Training started");
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed to start");
    } finally {
      setBusyName("");
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal text-zinc-950">Person Management</h1>
          <p className="text-sm text-zinc-500">{persons.length} registered persons</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <div className="inline-flex w-fit items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-600">
            <Users size={16} aria-hidden="true" />
            Active {persons.filter((person) => !isTruthy(person.blocked)).length}
          </div>
          <button type="button" disabled={busyName === "__train__"} onClick={runTraining} className="inline-flex h-10 items-center gap-2 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50">
            <RefreshCw size={16} aria-hidden="true" />
            Train
          </button>
        </div>
      </div>

      {error ? <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}
      {notice ? <div className="rounded-md border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">{notice}</div> : null}

      <section className="rounded-lg border border-zinc-200 bg-white">
        <div className="table-scroll">
          <table className="min-w-full text-left text-sm">
            <thead className="bg-zinc-50 text-xs uppercase text-zinc-500">
              <tr>
                <th className="px-4 py-3 font-semibold">Name</th>
                <th className="px-4 py-3 font-semibold">Registered</th>
                <th className="px-4 py-3 font-semibold">Attendances</th>
                <th className="px-4 py-3 font-semibold">Proxy Alerts</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {persons.map((person) => (
                <tr key={person.name} className="hover:bg-zinc-50">
                  <td className="px-4 py-3 font-medium text-zinc-950">{person.name}</td>
                  <td className="px-4 py-3 text-zinc-600">{formatDateTime(person.registered_at)}</td>
                  <td className="px-4 py-3 text-zinc-600">{person.total_attendances || 0}</td>
                  <td className="px-4 py-3 text-zinc-600">{person.proxy_count || 0}</td>
                  <td className="px-4 py-3"><PersonStatus blocked={person.blocked} /></td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-2">
                      <button type="button" disabled={busyName === person.name} onClick={() => runPersonAction(person.name, () => reenablePerson(person.name))} className="inline-flex h-9 items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50">
                        <RotateCcw size={15} aria-hidden="true" />
                        Re-enable
                      </button>
                      <button type="button" disabled={busyName === person.name} onClick={() => runPersonAction(person.name, () => blockPerson(person.name))} className="inline-flex h-9 items-center gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 text-sm font-medium text-amber-700 transition hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-50">
                        <LockKeyhole size={15} aria-hidden="true" />
                        Block
                      </button>
                      <button type="button" disabled={busyName === person.name} onClick={() => runPersonAction(person.name, () => unblockPerson(person.name))} className="inline-flex h-9 items-center gap-2 rounded-md border border-emerald-200 bg-emerald-50 px-3 text-sm font-medium text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50">
                        <UnlockKeyhole size={15} aria-hidden="true" />
                        Unblock
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {persons.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-sm text-zinc-500">No registered persons found</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
