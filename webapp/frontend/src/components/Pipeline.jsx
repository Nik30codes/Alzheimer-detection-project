// Fallback wording only. The server sends its own label for every stage — including the
// per-slice "Running model on slice 7 of 32" — and that always wins, because it reports
// what actually happened rather than what was planned.
const STAGES = [
  'Reading files',
  'Validating axial slices',
  'Harmonising resolution',
  'Running model + TTA on each slice',
  'Computing Grad-CAM',
  'Complete',
]

/**
 * Live readout of the analysis pipeline.
 *
 * Stages arrive from the server as each one actually finishes, so this reflects real
 * work rather than a timer. `labels` carries the server's own wording (e.g. the decoded
 * image dimensions), which is more informative than the static text.
 */
export default function Pipeline({ pct, current, labels }) {
  const running = current > 0 && current <= STAGES.length
  return (
    <div className="glass-soft rounded-2xl p-4 mt-5">
      <div className="flex items-baseline justify-between mb-3">
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-muted">
          Inference pipeline
        </span>
        <span className="font-mono text-sm text-accent2 tabular-nums">{pct}%</span>
      </div>

      <div className="relative h-1.5 rounded-full bg-line overflow-hidden">
        <div
          className={`relative h-full rounded-full transition-[width] duration-500 ease-out ${
            running && pct < 100 ? 'shimmer' : ''
          }`}
          style={{
            width: `${pct}%`,
            background:
              'linear-gradient(90deg,var(--color-accent),var(--color-accent3))',
            boxShadow: '0 0 14px rgb(245 158 11 / .5)',
            overflow: 'hidden',
          }}
        />
      </div>

      <ol className="mt-4 space-y-0.5">
        {STAGES.map((s, i) => {
          const n = i + 1
          const done = n < current
          const active = n === current
          return (
            <li
              key={s}
              className={`flex items-start gap-2.5 py-1 text-[13.5px] transition-colors ${
                done ? 'text-ok' : active ? 'text-ink' : 'text-muted/55'
              }`}
            >
              <span className="mt-[3px] grid h-4 w-4 shrink-0 place-items-center">
                {done ? (
                  <svg viewBox="0 0 16 16" className="h-4 w-4 fill-none stroke-current" strokeWidth="2.2">
                    <path d="M3 8.5l3.2 3.2L13 5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : active ? (
                  <span className="h-2 w-2 rounded-full bg-accent2 animate-pulse shadow-[0_0_10px_var(--color-accent2)]" />
                ) : (
                  <span className="h-1.5 w-1.5 rounded-full bg-line" />
                )}
              </span>
              <span className={active ? 'font-medium' : ''}>{labels[n] || s}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
