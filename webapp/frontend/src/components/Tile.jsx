/**
 * One metric tile. `accent` switches the value colour for status-bearing numbers so a
 * reader can scan the row without reading every label.
 */
export default function Tile({ label, value, sub, accent = 'grad', mono = false }) {
  const valueClass = {
    grad: 'grad-accent',
    ok: 'text-ok',
    warn: 'text-warn',
    plain: 'text-ink',
  }[accent]

  return (
    <div className="glass-soft topline rounded-2xl p-4 transition hover:border-accent/30">
      <div className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-muted">
        {label}
      </div>
      <div
        className={`${valueClass} ${mono ? 'font-mono' : ''} text-[27px] font-extrabold leading-none mt-2.5 mb-1.5 tabular-nums`}
      >
        {value}
      </div>
      {sub && <div className="text-[11.5px] leading-snug text-muted">{sub}</div>}
    </div>
  )
}
