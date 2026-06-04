"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Camera, RotateCcw, Square } from "lucide-react";
import { VerifyFrameResponse, verifyFrame } from "@/lib/api";
import { formatPercent } from "@/lib/format";

function stageClass(stage?: string) {
  switch (stage) {
    case "SUCCESS":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "BLOCKED":
      return "border-amber-200 bg-amber-50 text-amber-700";
    case "RETRY":
      return "border-sky-200 bg-sky-50 text-sky-700";
    case "BEHAVIOR":
      return "border-indigo-200 bg-indigo-50 text-indigo-700";
    case "BLINK":
      return "border-violet-200 bg-violet-50 text-violet-700";
    default:
      return "border-zinc-200 bg-zinc-50 text-zinc-700";
  }
}

function progressFor(result: VerifyFrameResponse | null) {
  if (!result) return 0;
  if (result.stage === "SUCCESS" || result.stage === "BLOCKED") return 100;
  if (result.stage === "BLINK") return Math.min(95, (result.elapsed_seconds / 5) * 100);
  if (result.stage === "BEHAVIOR") return Math.min(95, (result.elapsed_seconds / 25) * 100);
  if (result.stage === "RETRY") return 35;
  return 10;
}

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}`;
}

export default function VerifyPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [result, setResult] = useState<VerifyFrameResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSessionId(newSessionId());
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!cameraActive || !sessionId) return;

    async function tick() {
      if (inFlightRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d");
      if (!context) return;

      context.drawImage(video, 0, 0, width, height);
      const frame = canvas.toDataURL("image/jpeg", 0.76);

      inFlightRef.current = true;
      try {
        const response = await verifyFrame(frame, sessionId);
        setResult(response);
        setError("");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Verification failed");
      } finally {
        inFlightRef.current = false;
      }
    }

    tick();
    const interval = window.setInterval(tick, 500);
    return () => window.clearInterval(interval);
  }, [cameraActive, sessionId]);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera unavailable");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraActive(false);
  }

  function resetSession() {
    setSessionId(newSessionId());
    setResult(null);
    setError("");
  }

  const progress = useMemo(() => progressFor(result), [result]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal text-zinc-950">Live Webcam Verification</h1>
          <p className="text-sm text-zinc-500">Session {sessionId ? sessionId.slice(0, 8) : "pending"}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={cameraActive ? stopCamera : startCamera} className="inline-flex h-10 items-center gap-2 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white transition hover:bg-zinc-800">
            {cameraActive ? <Square size={16} aria-hidden="true" /> : <Camera size={16} aria-hidden="true" />}
            {cameraActive ? "Stop" : "Start"}
          </button>
          <button type="button" onClick={resetSession} className="inline-flex h-10 items-center gap-2 rounded-md border border-zinc-200 bg-white px-3 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50">
            <RotateCcw size={16} aria-hidden="true" />
            Reset
          </button>
        </div>
      </div>

      {error ? <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}

      <section className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
        <div className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-950">
          <div className="relative aspect-video min-h-80">
            <video ref={videoRef} autoPlay playsInline muted className="h-full w-full scale-x-[-1] object-cover" />
            {!cameraActive ? <div className="absolute inset-0 flex items-center justify-center bg-zinc-950 text-sm font-medium text-zinc-300">Camera offline</div> : null}
            <div className="absolute left-4 top-4">
              <span className={`inline-flex items-center rounded-md border px-3 py-1 text-xs font-semibold ${stageClass(result?.stage)}`}>{result?.stage || "FACE"}</span>
            </div>
            {result?.stage === "SUCCESS" || result?.stage === "RETRY" || result?.stage === "BLOCKED" ? (
              <div className="absolute inset-x-4 bottom-4 rounded-lg border border-white/20 bg-white/95 p-4 shadow-sm">
                <p className="text-base font-semibold text-zinc-950">{result.message}</p>
                <p className="mt-1 text-sm text-zinc-600">{result.person_name || "Unknown"}</p>
              </div>
            ) : null}
          </div>
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <div className="space-y-4">
          <section className="rounded-lg border border-zinc-200 bg-white p-4">
            <h2 className="text-base font-semibold tracking-normal text-zinc-950">Stage</h2>
            <div className="mt-4 h-3 overflow-hidden rounded-md bg-zinc-100">
              <div className="h-full rounded-md bg-zinc-950 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div><p className="text-zinc-500">Person</p><p className="mt-1 font-medium text-zinc-950">{result?.person_name || "-"}</p></div>
              <div><p className="text-zinc-500">Attempt</p><p className="mt-1 font-medium text-zinc-950">{result?.attempt ?? 0}</p></div>
              <div><p className="text-zinc-500">Face</p><p className="mt-1 font-medium text-zinc-950">{formatPercent(result?.face_confidence)}</p></div>
              <div><p className="text-zinc-500">Behavior</p><p className="mt-1 font-medium text-zinc-950">{formatPercent(result?.behavior_confidence)}</p></div>
            </div>
          </section>

          <section className="rounded-lg border border-zinc-200 bg-white p-4">
            <h2 className="text-base font-semibold tracking-normal text-zinc-950">Decision</h2>
            <p className="mt-3 text-sm text-zinc-600">{result?.message || "Waiting"}</p>
            <p className="mt-3 text-sm font-medium text-zinc-950">{result?.decision || "PENDING"}</p>
          </section>
        </div>
      </section>
    </div>
  );
}
