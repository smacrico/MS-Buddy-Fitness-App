import { useEffect, useState } from "react";

type Summary = {
  data_points: number;
  date_range: { start: string | null; end: string | null };
  current_values: Record<string, number>;
  recovery_scores: { ms: number };
  baselines: Record<string, number>;
  alerts: Array<Record<string, unknown>>;
  anomalies: Array<Record<string, unknown>>;
};

type SyncResult = {
  source_view: string;
  imported_count: number;
  source_name_used: string;
  db_path: string;
  analysis_triggered: boolean;
};

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [status, setStatus] = useState("connecting");
  const [message, setMessage] = useState("");

  async function loadSummary() {
    const resp = await fetch("/api/summary");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    setSummary(await resp.json());
  }

  async function syncArtemis() {
    setMessage("Syncing Artemis...");
    const resp = await fetch("/api/import/artemis", { method: "POST" });
    if (!resp.ok) {
      setMessage(`Sync failed: HTTP ${resp.status}`);
      return;
    }
    const result: SyncResult = await resp.json();
    setMessage(
      `Imported ${result.imported_count} rows from ${result.source_view} at ${result.db_path}`
    );
    await loadSummary();
  }

  useEffect(() => {
    loadSummary().catch((err) => {
      console.error("Failed to load summary:", err);
      setStatus("api error");
    });

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/live`);

    ws.onopen = () => setStatus("live");
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("ws error");

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "summary_updated" && msg.payload) {
          setSummary(msg.payload);
        }
        if (msg.type === "artemis_synced" && msg.payload) {
          setMessage(
            `Imported ${msg.payload.imported_count} rows from ${msg.payload.source_view}`
          );
        }
        if (msg.type === "artemis_sync_error" && msg.payload) {
          setMessage(`Artemis sync error: ${msg.payload.message}`);
        }
      } catch (e) {
        console.error("Bad websocket message:", e);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <main style={{ fontFamily: "Arial, sans-serif", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>HRV Platform</h1>
      <p>Connection: {status}</p>
      <button onClick={syncArtemis}>Sync from Artemis</button>
      <p>{message}</p>

      {!summary ? (
        <p>Loading...</p>
      ) : (
        <>
          <section>
            <h2>Summary</h2>
            <p>Data points: {summary.data_points}</p>
            <p>MS score: {summary.recovery_scores?.ms ?? "n/a"}</p>
            <p>
              Date range: {summary.date_range?.start ?? "n/a"} → {summary.date_range?.end ?? "n/a"}
            </p>
          </section>

          <section>
            <h2>Current values</h2>
            <pre>{JSON.stringify(summary.current_values, null, 2)}</pre>
          </section>

          <section>
            <h2>Baselines</h2>
            <pre>{JSON.stringify(summary.baselines ?? {}, null, 2)}</pre>
          </section>

          <section>
            <h2>Alerts</h2>
            <pre>{JSON.stringify(summary.alerts ?? [], null, 2)}</pre>
          </section>

          <section>
            <h2>Anomalies</h2>
            <pre>{JSON.stringify(summary.anomalies ?? [], null, 2)}</pre>
          </section>
        </>
      )}
    </main>
  );
}
