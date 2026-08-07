import { useParams } from "react-router-dom";

export default function ReportViewer() {
  const { scanId } = useParams();
  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold">Report</h1>
      <p className="text-gray-400">Forensic report for scan {scanId}.</p>
    </div>
  );
}
