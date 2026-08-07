export interface ModelResult {
  model_name: string;
  score: number | null;
  confidence: number | null;
  metadata: Record<string, unknown> | null;
}

export interface ScanResponse {
  scan_id: string;
  status: "pending" | "processing" | "completed" | "failed" | "blocked_on_model_integration";
  media_type: "image" | "video";
  fused_score: number | null;
  fused_verdict: string | null;
  explanation: string | null;
  error: string | null;
  model_results: ModelResult[];
}

const TERMINAL_STATUSES = new Set(["completed", "failed", "blocked_on_model_integration"]);

export async function createScan(file: File): Promise<{ scan_id: string; status: string }> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/v1/scan/", { method: "POST", body });
  if (!res.ok) throw new Error(`scan upload failed: ${res.status}`);
  return res.json();
}

export async function getScan(scanId: string): Promise<ScanResponse> {
  const res = await fetch(`/api/v1/scan/${scanId}`);
  if (!res.ok) throw new Error(`scan fetch failed: ${res.status}`);
  return res.json();
}

export function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export async function getExplanation(scanId: string): Promise<{ explanation: string | null; status: string }> {
  const res = await fetch(`/api/v1/explain/${scanId}`);
  if (res.status === 501) {
    return { explanation: null, status: "blocked_on_model_integration" };
  }
  if (!res.ok) throw new Error(`explanation fetch failed: ${res.status}`);
  return res.json();
}

export function heatmapUrl(scanId: string): string {
  return `/api/v1/scan/${scanId}/heatmap`;
}
