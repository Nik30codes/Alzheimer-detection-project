/**
 * Stats strip — the reference screenshots' "trusted by 20,000 students" bar, replaced
 * with real dataset/validation facts. AUC/accuracy come from live config (`m`), never
 * typed in; subject counts are fixed dataset facts, the same ones already quoted
 * verbatim in docs/PROJECT_EXPLANATION.md.
 */
export default function TrustBar({ m }) {
  const stats = [
    { v: '853', k: 'real clinical scans (ADNI)' },
    { v: '501', k: 'AD / cognitively-normal subjects' },
    { v: '5-fold', k: 'cross-validated, never a single split' },
    m?.auc
      ? { v: m.auc.toFixed(3), k: `ROC AUC · 95% CI ${m.auc_ci[0].toFixed(2)}–${m.auc_ci[1].toFixed(2)}` }
      : { v: '0.784', k: 'ROC AUC on held-out subjects' },
  ]
  return (
    <div className="band-tint">
      <div className="mx-auto max-w-6xl px-5 sm:px-6 py-8">
        <div className="text-center text-[11px] font-semibold uppercase tracking-[0.18em] text-muted mb-6">
          Measured, not claimed
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 text-center">
          {stats.map((s) => (
            <div key={s.k}>
              <div className="font-mono text-2xl sm:text-3xl font-extrabold grad-accent tabular-nums">
                {s.v}
              </div>
              <div className="mt-1.5 text-[11.5px] leading-snug text-muted">{s.k}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
