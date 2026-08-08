"""Forensic report generation (PDF) from a fused verdict + explanation, with signing.

One PDF per scan: fused verdict, every model's individual score, the
Grad-CAM heatmap (when available), the narrative explanation, and an
HMAC signature over the verdict so the report can be verified as
untampered later. Kept to plain reportlab primitives -- no HTML/CSS
renderer dependency, no system packages beyond the pure-Python wheel.
"""
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.db_models import Scan, ScanResult
from app.reports.sign import sign_report
from app.storage import download_media

INK = HexColor("#141414")
MUTED = HexColor("#6b6b6b")
AMBER = HexColor("#c8862b")
LINE = HexColor("#d8d4cc")

MODEL_LABELS: dict[str, str] = {
    "frame_classifier": "Visual (frame-level)",
    "temporal_classifier": "Visual (temporal)",
    "univfd": "Visual (UnivFD)",
    "audio_deepfake": "Audio (cloned-voice)",
    "lipsync": "Lip-sync consistency",
    "liveness": "Liveness / anti-spoof",
    "face_recognition": "Face recognition",
    "device": "Device / capture fingerprint",
}


def _canonical_payload(scan: Scan) -> bytes:
    created = scan.created_at.isoformat() if scan.created_at else ""
    fields = [scan.id, scan.status, scan.fused_verdict or "", str(scan.fused_score), created]
    return "|".join(fields).encode()


def _heatmap_png(results: list[ScanResult]) -> bytes | None:
    gradcam = next((r for r in results if r.model_name == "gradcam"), None)
    heatmap_key = (gradcam.result_metadata or {}).get("heatmap_key") if gradcam else None
    if not heatmap_key:
        return None
    return download_media(heatmap_key)


def _wrap(text: str, width_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width_chars and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def generate_report(scan: Scan, results: list[ScanResult]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    y = height - margin

    def rule(gap_before: float = 10, gap_after: float = 14) -> None:
        nonlocal y
        y -= gap_before
        c.setStrokeColor(LINE)
        c.line(margin, y, width - margin, y)
        y -= gap_after

    # Header
    c.setFillColor(AMBER)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "MORPHEUS.AI")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawRightString(width - margin, y, "FORENSIC REPORT")
    y -= 22
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, "Deepfake Detection Verdict")
    y -= 16
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c.drawString(margin, y, f"Scan ID: {scan.id}    Media: {scan.media_type}    Generated: {generated_at}")
    rule()

    # Fused verdict
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Fused Verdict")
    y -= 18
    verdict = (scan.fused_verdict or "UNKNOWN").upper()
    score_pct = f"{scan.fused_score * 100:.1f}%" if scan.fused_score is not None else "N/A"
    verdict_color = colors.red if verdict in ("FAKE", "SPOOF", "SYNTHETIC") else colors.HexColor("#2f7d4f")
    c.setFillColor(verdict_color)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, y, verdict)
    c.setFillColor(INK)
    c.setFont("Helvetica", 14)
    c.drawRightString(width - margin, y + 4, f"confidence {score_pct}")
    rule(gap_before=20)

    # Per-model breakdown
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Signal Breakdown")
    y -= 18
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(MUTED)
    c.drawString(margin, y, "CHANNEL")
    c.drawString(margin + 260, y, "SCORE")
    c.drawString(margin + 340, y, "CONFIDENCE")
    y -= 6
    c.setStrokeColor(LINE)
    c.line(margin, y, width - margin, y)
    y -= 14

    reportable = [r for r in results if r.model_name in MODEL_LABELS]
    for r in reportable:
        c.setFont("Helvetica", 10)
        c.setFillColor(INK)
        c.drawString(margin, y, MODEL_LABELS.get(r.model_name, r.model_name))
        score_str = f"{r.score * 100:.1f}%" if r.score is not None else "not run"
        conf_str = f"{r.confidence * 100:.1f}%" if r.confidence is not None else "—"
        c.drawString(margin + 260, y, score_str)
        c.drawString(margin + 340, y, conf_str)
        y -= 16
    if not reportable:
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(MUTED)
        c.drawString(margin, y, "No per-channel results recorded for this scan.")
        y -= 16
    rule(gap_before=6)

    # Grad-CAM heatmap
    heatmap_bytes = _heatmap_png(results)
    if heatmap_bytes:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Grad-CAM Heatmap")
        y -= 12
        img = ImageReader(io.BytesIO(heatmap_bytes))
        img_w, img_h = img.getSize()
        draw_w = 2.2 * inch
        draw_h = draw_w * img_h / img_w
        c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
        y -= draw_h + 14
        rule(gap_before=6)

    # Narrative explanation
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Narrative Explanation")
    y -= 16
    c.setFont("Helvetica", 10)
    c.setFillColor(INK)
    explanation = scan.explanation or "No narrative explanation was generated for this scan."
    for line in _wrap(explanation, 95):
        if y < margin + 60:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        c.drawString(margin, y, line)
        y -= 13

    # Signature footer
    payload = _canonical_payload(scan)
    signature = sign_report(payload).decode()
    if y < margin + 60:
        c.showPage()
        y = height - margin
    y = max(y, margin + 50)
    c.setStrokeColor(LINE)
    c.line(margin, y, width - margin, y)
    y -= 14
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(margin, y, "Signature (HMAC-SHA256 over scan_id|status|verdict|score|created_at):")
    y -= 10
    c.setFont("Courier", 7)
    c.drawString(margin, y, signature)

    c.showPage()
    c.save()
    return buf.getvalue()
