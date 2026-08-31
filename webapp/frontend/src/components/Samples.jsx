import { useEffect, useState } from 'react'

/**
 * One-click demo scans.
 *
 * Without these the page is unusable by anyone who does not happen to have a brain MRI
 * on their laptop — which is nearly every visitor. Samples come from the held-out TEST
 * split, so they are images the model never trained on: a correct answer here is a fair
 * demonstration rather than a memorised one.
 *
 * Two tiers, and the distinction matters:
 *   * SINGLE slices — one image, the noisier task the published figures do NOT describe.
 *   * FULL SUBJECTS — all 32 axial slices, fetched and posted together so the backend
 *     averages them. This is the input the reported accuracy and ROC AUC were actually
 *     computed on, so it is the only option that reproduces the headline number.
 */
export default function Samples({ onPick, disabled }) {
  const [samples, setSamples] = useState([])
  const [active, setActive] = useState(null)
  const [loading, setLoading] = useState(null)

  useEffect(() => {
    fetch('/samples/index.json')
      .then((r) => (r.ok ? r.json() : []))
      .then((s) => setSamples(Array.isArray(s) ? s : []))
      .catch(() => setSamples([]))
  }, [])

  if (!samples.length) return null

  // Older sample indexes have no `multi`/`files` fields, so fall back to `file`.
  const filesOf = (s) => s.files || [s.file]
  const singles = samples.filter((s) => !s.multi)
  const subjects = samples.filter((s) => s.multi)

  async function choose(s) {
    const paths = filesOf(s)
    setLoading(s.file)
    try {
      const blobs = await Promise.all(
        paths.map((p) => fetch(p).then((r) => r.blob())),
      )
      setActive(s.file)
      onPick(
        blobs.map((b, i) => new File([b], paths[i].split('/').pop(), { type: 'image/png' })),
        s,
      )
    } finally {
      setLoading(null)
    }
  }

  const Thumb = ({ s, wide }) => {
    const on = active === s.file
    const busy = loading === s.file
    return (
      <button
        type="button"
        disabled={disabled || busy}
        onClick={() => choose(s)}
        className={`group relative overflow-hidden rounded-2xl border p-2 text-left transition disabled:opacity-40 ${
          wide ? 'w-[168px]' : 'w-[112px]'
        } ${
          on
            ? 'border-accent2 bg-accent/12'
            : 'border-line bg-black/[0.015] hover:border-accent2/60 hover:bg-black/[0.03]'
        }`}
      >
        <div className="relative">
          <img
            src={s.file}
            alt={s.label}
            className="mb-2 block w-full rounded-xl bg-black transition group-hover:brightness-110"
          />
          {s.multi && (
            <span
              className="absolute left-1.5 top-1.5 rounded-full px-2 py-0.5 text-[9.5px] font-bold uppercase tracking-wider text-white"
              style={{
                background:
                  'linear-gradient(100deg,var(--color-accent2),var(--color-accent3))',
              }}
            >
              {s.n_slices} slices
            </span>
          )}
          {busy && (
            <span className="absolute inset-0 grid place-items-center rounded-xl bg-bg0/85 text-[11px] font-semibold text-accent2">
              loading…
            </span>
          )}
        </div>
        <div className="flex items-center justify-between px-0.5">
          <span className="text-[10px] uppercase tracking-wider text-muted">truth</span>
          <span
            className={`font-mono text-[11.5px] font-bold ${
              s.true_class === 'AD' ? 'text-warn' : 'text-ok'
            }`}
          >
            {s.true_class}
          </span>
        </div>
      </button>
    )
  }

  return (
    <div className="mt-6 space-y-6">
      {subjects.length > 0 && (
        <div>
          <div className="mb-3 flex flex-wrap items-baseline gap-2">
            <span className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-accent2">
              Try a full 32-slice subject
            </span>
            <span className="text-[12px] text-muted/70">
              reproduces the published subject-level metric
            </span>
          </div>
          <div className="flex flex-wrap gap-3">
            {subjects.map((s) => (
              <Thumb key={s.file} s={s} wide />
            ))}
          </div>
          <p className="mt-3 max-w-2xl text-[11.5px] leading-relaxed text-muted">
            Every axial slice for one held-out subject is sent at once. Each is scored
            separately and the probabilities are averaged into one prediction — the same
            aggregation used to compute the reported accuracy and ROC AUC.
          </p>
        </div>
      )}

      {singles.length > 0 && (
        <div>
          <div className="mb-3 flex flex-wrap items-baseline gap-2">
            <span className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-muted">
              Or a single slice
            </span>
            <span className="text-[12px] text-muted/70">the harder, noisier task</span>
          </div>
          <div className="flex flex-wrap gap-3">
            {singles.map((s) => (
              <Thumb key={s.file} s={s} />
            ))}
          </div>
          <p className="mt-3 max-w-2xl text-[11.5px] leading-relaxed text-muted">
            Held-out subjects the model never trained on. One slice on its own runs several
            points below the headline figure, which averages 32 slices per person — so a
            miss here is expected rather than hidden.
          </p>
        </div>
      )}
    </div>
  )
}
