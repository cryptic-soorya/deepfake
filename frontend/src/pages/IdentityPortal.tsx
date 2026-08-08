import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import clsx from "clsx";
import CornerFrame from "../components/hud/CornerFrame";
import { enrollIdentity, verifyIdentity, type VerifyResponse } from "../lib/api";

const STEPS = [
  { num: "01", label: "Capture", desc: "Face detection + landmark alignment (SCRFD)" },
  { num: "02", label: "Liveness", desc: "Anti-spoofing pass (Silent-Face / MiniFASNet)" },
  { num: "03", label: "Match", desc: "Identity embedding + comparison (ArcFace)" },
];

type EnrollState = { status: "idle" | "submitting" | "done" | "error"; message?: string };
type VerifyState = { status: "idle" | "submitting" | "done" | "error"; result?: VerifyResponse; message?: string };

function ConfidencePill({ ok, okLabel, notOkLabel, confidence }: { ok: boolean; okLabel: string; notOkLabel: string; confidence: number }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-2 border px-3 py-1 font-mono text-xs uppercase tracking-widest",
        ok ? "border-signal/40 bg-signal/10 text-signal" : "border-alert/40 bg-alert/10 text-alert",
      )}
    >
      <span className={clsx("h-1.5 w-1.5 rounded-full", ok ? "bg-signal" : "bg-alert")} />
      {ok ? okLabel : notOkLabel}
      <span className="text-bone/50">/</span>
      {(confidence * 100).toFixed(1)}% confidence
    </span>
  );
}

function useImagePicker() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFiles = useCallback((files: FileList | null) => {
    if (files && files[0]) setFile(files[0]);
  }, []);

  return { file, previewUrl, dragging, setDragging, inputRef, handleFiles };
}

export default function IdentityPortal() {
  const [userId, setUserId] = useState("");
  const enrollPicker = useImagePicker();
  const verifyPicker = useImagePicker();
  const [enroll, setEnroll] = useState<EnrollState>({ status: "idle" });
  const [verify, setVerify] = useState<VerifyState>({ status: "idle" });

  const handleEnroll = useCallback(async () => {
    if (!enrollPicker.file || !userId.trim() || enroll.status === "submitting") return;
    setEnroll({ status: "submitting" });
    try {
      const result = await enrollIdentity(userId.trim(), enrollPicker.file);
      setEnroll({ status: "done", message: `Enrolled — detector confidence ${(result.det_score * 100).toFixed(1)}%` });
    } catch (err) {
      setEnroll({ status: "error", message: err instanceof Error ? err.message : "enrollment failed" });
    }
  }, [enrollPicker.file, userId, enroll.status]);

  const handleVerify = useCallback(async () => {
    if (!verifyPicker.file || !userId.trim() || verify.status === "submitting") return;
    setVerify({ status: "submitting" });
    try {
      const result = await verifyIdentity(userId.trim(), verifyPicker.file);
      setVerify({ status: "done", result });
    } catch (err) {
      setVerify({ status: "error", message: err instanceof Error ? err.message : "verification failed" });
    }
  }, [verifyPicker.file, userId, verify.status]);

  const canEnroll = Boolean(enrollPicker.file && userId.trim()) && enroll.status !== "submitting";
  const canVerify = Boolean(verifyPicker.file && userId.trim()) && verify.status !== "submitting";

  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-xs uppercase tracking-[0.3em] text-amber/70"
      >
        Module 03 // Identity
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-2 font-display text-4xl font-semibold sm:text-5xl"
      >
        Identity Portal
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-3 max-w-xl font-mono text-sm text-bone/50"
      >
        Enroll and verify a person's identity via face match and liveness — confirms the person behind the pixels is real and present.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-6 sm:grid-cols-[1fr_1.2fr]"
      >
        <CornerFrame active={enrollPicker.dragging}>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              enrollPicker.setDragging(true);
            }}
            onDragLeave={() => enrollPicker.setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              enrollPicker.setDragging(false);
              enrollPicker.handleFiles(e.dataTransfer.files);
            }}
            onClick={() => enrollPicker.inputRef.current?.click()}
            className="relative flex min-h-[20rem] cursor-pointer items-center justify-center overflow-hidden p-6 text-center"
          >
            <input
              ref={enrollPicker.inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => enrollPicker.handleFiles(e.target.files)}
            />
            {enrollPicker.previewUrl && (
              <img
                src={enrollPicker.previewUrl}
                alt="Selected portrait"
                className="absolute inset-0 h-full w-full object-cover"
              />
            )}
            <div className="relative flex h-48 w-48 items-center justify-center">
              <span className="absolute -top-1 -left-1 h-6 w-6 border-l-2 border-t-2 border-amber" />
              <span className="absolute -top-1 -right-1 h-6 w-6 border-r-2 border-t-2 border-amber" />
              <span className="absolute -bottom-1 -left-1 h-6 w-6 border-l-2 border-b-2 border-amber" />
              <span className="absolute -bottom-1 -right-1 h-6 w-6 border-r-2 border-b-2 border-amber" />
              {!enrollPicker.previewUrl && (
                <span className="font-mono text-[10px] uppercase tracking-widest text-bone/30">Align Face</span>
              )}
            </div>
            {enrollPicker.file && (
              <p className="absolute inset-x-0 bottom-0 bg-void/70 px-3 py-1.5 font-mono text-[11px] text-bone/80">
                {enrollPicker.file.name}
              </p>
            )}
          </div>
        </CornerFrame>

        <div className="flex flex-col gap-4">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.num}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 + i * 0.08 }}
              className="flex items-start gap-4 border border-line bg-ink/40 p-4"
            >
              <span className="font-mono text-lg text-amber/60">{s.num}</span>
              <div>
                <p className="font-display text-sm font-semibold text-bone">{s.label}</p>
                <p className="mt-1 font-mono text-xs text-bone/40">{s.desc}</p>
              </div>
            </motion.div>
          ))}

          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="user id"
            className="border border-line bg-ink/40 px-4 py-2.5 font-mono text-xs text-bone placeholder:text-bone/30 focus:border-amber focus:outline-none"
          />

          <button
            disabled={!canEnroll}
            onClick={handleEnroll}
            className="mt-2 border border-amber bg-amber/10 px-6 py-2.5 font-mono text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void disabled:cursor-not-allowed disabled:border-line-bright disabled:bg-transparent disabled:text-bone/20"
          >
            {enroll.status === "submitting" ? "Enrolling…" : "Begin Enrollment →"}
          </button>

          {enroll.status !== "idle" && (
            <p className={`font-mono text-xs ${enroll.status === "error" ? "text-red-400" : "text-bone/50"}`}>
              {enroll.message}
            </p>
          )}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mt-6 grid gap-6 sm:grid-cols-[1fr_1.2fr]"
      >
        <CornerFrame active={verifyPicker.dragging}>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              verifyPicker.setDragging(true);
            }}
            onDragLeave={() => verifyPicker.setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              verifyPicker.setDragging(false);
              verifyPicker.handleFiles(e.dataTransfer.files);
            }}
            onClick={() => verifyPicker.inputRef.current?.click()}
            className="relative flex min-h-[16rem] cursor-pointer items-center justify-center overflow-hidden p-6 text-center"
          >
            <input
              ref={verifyPicker.inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => verifyPicker.handleFiles(e.target.files)}
            />
            {verifyPicker.previewUrl && (
              <img
                src={verifyPicker.previewUrl}
                alt="Probe portrait"
                className="absolute inset-0 h-full w-full object-cover"
              />
            )}
            <div className="relative flex h-40 w-40 items-center justify-center">
              <span className="absolute -top-1 -left-1 h-6 w-6 border-l-2 border-t-2 border-amber" />
              <span className="absolute -top-1 -right-1 h-6 w-6 border-r-2 border-t-2 border-amber" />
              <span className="absolute -bottom-1 -left-1 h-6 w-6 border-l-2 border-b-2 border-amber" />
              <span className="absolute -bottom-1 -right-1 h-6 w-6 border-r-2 border-b-2 border-amber" />
              {!verifyPicker.previewUrl && (
                <span className="font-mono text-[10px] uppercase tracking-widest text-bone/30">Probe Face</span>
              )}
            </div>
            {verifyPicker.file && (
              <p className="absolute inset-x-0 bottom-0 bg-void/70 px-3 py-1.5 font-mono text-[11px] text-bone/80">
                {verifyPicker.file.name}
              </p>
            )}
          </div>
        </CornerFrame>

        <div className="flex flex-col gap-4">
          <div className="border border-line bg-ink/40 p-4">
            <p className="font-display text-sm font-semibold text-bone">Verify Identity</p>
            <p className="mt-1 font-mono text-xs text-bone/40">
              Submit a live probe photo for the same user id — checks it's a real, present person (liveness) and
              that the face matches the enrolled template (ArcFace).
            </p>
          </div>

          <button
            disabled={!canVerify}
            onClick={handleVerify}
            className="border border-amber bg-amber/10 px-6 py-2.5 font-mono text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void disabled:cursor-not-allowed disabled:border-line-bright disabled:bg-transparent disabled:text-bone/20"
          >
            {verify.status === "submitting" ? "Verifying…" : "Verify Identity →"}
          </button>

          {verify.status === "error" && <p className="font-mono text-xs text-red-400">{verify.message}</p>}

          {verify.status === "done" && verify.result && (
            <div className="flex flex-col gap-3">
              <ConfidencePill
                ok={verify.result.verified}
                okLabel="Verified"
                notOkLabel="Not Verified"
                confidence={
                  verify.result.verified
                    ? Math.min(verify.result.liveness.score, verify.result.match_score)
                    : 1 - Math.min(verify.result.liveness.score, verify.result.match_score)
                }
              />
              <ConfidencePill
                ok={verify.result.liveness.is_live}
                okLabel="Live"
                notOkLabel="Spoof"
                confidence={
                  verify.result.liveness.is_live ? verify.result.liveness.score : 1 - verify.result.liveness.score
                }
              />
              <ConfidencePill
                ok={verify.result.is_match}
                okLabel="Match"
                notOkLabel="No Match"
                confidence={verify.result.match_score}
              />
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
