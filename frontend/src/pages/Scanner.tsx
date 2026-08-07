import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import CornerFrame from "../components/hud/CornerFrame";
import { createScan } from "../lib/api";

export default function Scanner() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleFiles = useCallback((files: FileList | null) => {
    if (files && files[0]) setFile(files[0]);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const { scan_id } = await createScan(file);
      navigate(`/report/${scan_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "upload failed");
      setSubmitting(false);
    }
  }, [file, submitting, navigate]);

  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-xs uppercase tracking-[0.3em] text-amber/70"
      >
        Module 01 // Scan
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-2 font-display text-4xl font-semibold sm:text-5xl"
      >
        Deepfake Scanner
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-3 max-w-xl font-mono text-sm text-bone/50"
      >
        Upload a video or image. Visual, audio, and lip-sync channels are scored independently and fused into one verdict.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10"
      >
        <CornerFrame active={dragging}>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => inputRef.current?.click()}
            className="relative flex min-h-[22rem] cursor-pointer flex-col items-center justify-center gap-4 overflow-hidden px-6 text-center"
          >
            {dragging && (
              <div className="pointer-events-none absolute inset-x-0 top-0 h-16 animate-scan bg-gradient-to-b from-amber/30 to-transparent" />
            )}
            <input
              ref={inputRef}
              type="file"
              accept="video/*,image/*"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            {file ? (
              <>
                <p className="font-mono text-sm text-bone">{file.name}</p>
                <p className="font-mono text-xs text-bone/40">{(file.size / 1024 / 1024).toFixed(2)} MB — ready for submission</p>
              </>
            ) : (
              <>
                <div className="flex h-14 w-14 items-center justify-center border border-line-bright font-mono text-amber">
                  &darr;
                </div>
                <p className="font-mono text-sm text-bone/70">Drop a file, or click to browse</p>
                <p className="font-mono text-xs text-bone/30">MP4, MOV, JPG, PNG — up to 200MB</p>
              </>
            )}
          </div>
        </CornerFrame>

        <div className="mt-6 flex items-center justify-between">
          <p className="font-mono text-xs text-bone/30">
            {error ?? "Analysis runs against the full detection stack — see the model table in the docs."}
          </p>
          <button
            disabled={!file || submitting}
            onClick={handleSubmit}
            className="border border-amber bg-amber/10 px-6 py-2.5 font-mono text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void disabled:cursor-not-allowed disabled:border-line-bright disabled:bg-transparent disabled:text-bone/20"
          >
            {submitting ? "Uploading…" : "Submit Scan →"}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
