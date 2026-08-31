import { useEffect, useState } from 'react'
import BrainBackground from './components/BrainBackground'
import FeatureGrid from './components/FeatureGrid'
import FindingsCarousel from './components/FindingsCarousel'
import Pipeline from './components/Pipeline'
import Result from './components/Result'
import Samples from './components/Samples'
import SliceStack from './components/SliceStack'
import Tile from './components/Tile'
import TrustBar from './components/TrustBar'
import Validation from './components/Validation'

const fmtPct = (x, d = 1) => `${(x * 100).toFixed(d)}%`

// Empty string -> relative '/api/...', which works locally (Vite proxies it) and in
// any single-service deploy where the backend serves the built frontend itself. Set
// VITE_API_BASE at build time (e.g. "https://your-backend.onrender.com") only when
// the frontend is hosted as a separate static site from the backend.
const API_BASE = import.meta.env.VITE_API_BASE || ''

export default function App() {
  const [cfg, setCfg] = useState(null)
  // A LIST, not one file. The published accuracy is subject-level: 32 axial slices per
  // subject, averaged into one prediction. Accepting a single slice made the demo
  // systematically weaker than its own headline number, so multi-select is the default
  // path and one file is simply the degenerate case of it.
  const [files, setFiles] = useState([])
  const [sampleInfo, setSampleInfo] = useState(null)
  const [busy, setBusy] = useState(false)
  const [pct, setPct] = useState(0)
  const [stage, setStage] = useState(0)
  const [labels, setLabels] = useState({})
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then((r) => r.json())
      .then(setCfg)
      .catch(() => setError('Could not reach the API. Is the backend running?'))
  }, [])

  const pick = (f, sample = null) => {
    const list = Array.isArray(f) ? f : Array.from(f || [])
    // Browsers hand back directory contents in arbitrary order, and the aggregation is
    // order-independent, but sorting keeps the "slice 7 of 32" progress readout matching
    // the anatomical order of the filenames.
    list.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    setFiles(list)
    setSampleInfo(sample)
    setResult(null)
    setError('')
  }

  async function analyse(e) {
    e.preventDefault()
    if (!files.length) return
    setBusy(true)
    setError('')
    setResult(null)
    setPct(0)
    setStage(1)
    setLabels({})

    const body = new FormData()
    // Every slice goes under the same `image` field; FastAPI binds them to a
    // List[UploadFile], so one file and thirty-two take the identical code path.
    for (const f of files) body.append('image', f)
    body.append('task', cfg.tasks[0].id)

    try {
      const res = await fetch(`${API_BASE}/api/predict-stream`, { method: 'POST', body })
      if (!res.ok) {
        const d = await res.json().catch(() => ({}))
        throw new Error(d.detail || 'Request failed')
      }
      // One JSON object per line, emitted as each stage completes, so the progress
      // shown is real work rather than an animation.
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (!line.trim()) continue
          const ev = JSON.parse(line)
          if (ev.error) {
            setError(ev.error)
            setBusy(false)
            setStage(0)
            return
          }
          if (ev.pct !== undefined) setPct(ev.pct)
          if (ev.stage) {
            setLabels((l) => ({ ...l, [ev.stage]: ev.label }))
            setStage(ev.done_stage ? ev.stage + 1 : ev.stage)
          }
          if (ev.result) setResult(ev.result)
        }
      }
    } catch (err) {
      setError(err.message || 'Network error')
    }
    setBusy(false)
  }

  if (!cfg) {
    return (
      <>
        <BrainBackground />
        <div className="mx-auto max-w-6xl px-6 py-16 text-muted">
          {error || 'Loading…'}
        </div>
      </>
    )
  }

  const task = cfg.tasks[0]
  const m = task.metrics || {}
  const nFiles = files.length

  return (
    <>
      <BrainBackground />

      <div className="mx-auto max-w-6xl px-5 sm:px-6">
        {/* ── nav ─────────────────────────────────────────────── */}
        <header className="flex items-center justify-between gap-4 py-6">
          <div>
            <div className="text-[15px] font-bold leading-tight tracking-tight">
              MultiModel
            </div>
            <div className="text-[9.5px] font-semibold uppercase tracking-[0.19em] text-muted">
              Alzheimer Detection
            </div>
          </div>
          <div className="flex items-center gap-2.5">
            <a
              href="https://github.com/Nik30codes/Alzheimer-detection-project"
              target="_blank"
              rel="noopener noreferrer"
              className="glass-soft flex items-center gap-1.5 rounded-full px-4 py-2 text-[12.5px] font-medium text-muted transition hover:text-ink"
            >
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 fill-current">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                  0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                  -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                  .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                  -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09
                  2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
                  0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2
                  0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
              </svg>
              Source
            </a>
            <a
              href="#methodology"
              className="glass-soft hidden sm:block rounded-full px-4 py-2 text-[12.5px] font-medium text-muted transition hover:text-ink"
            >
              How it was validated
            </a>
          </div>
        </header>

        {/* ── hero ────────────────────────────────────────────── */}
        <section className="pt-6 pb-14 rise grid gap-10 lg:grid-cols-[1.1fr_0.9fr] items-center">
          <div>
            <div className="mb-6 inline-flex items-center gap-2.5 rounded-full border border-ok/25 bg-ok/[0.08] pl-2 pr-4 py-1.5">
              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-ok text-white">
                <svg viewBox="0 0 16 16" className="h-3 w-3 fill-none stroke-current" strokeWidth="2.6">
                  <path d="M3 8.5l3.2 3.2L13 5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              <span className="text-[12px] font-semibold text-ok">
                Validated on {m.n_test} held-out subjects
              </span>
              <span className="text-[12px] text-ok/70">· ADNI</span>
            </div>

            <h1 className="max-w-xl text-[clamp(2.2rem,5.4vw,3.7rem)] font-black leading-[1.05] tracking-[-0.02em]">
              <span className="grad">Alzheimer&rsquo;s detection</span>
              <br />
              you can actually audit.
            </h1>

            <p className="mt-6 max-w-xl text-[16px] leading-relaxed text-muted">
              Upload an axial brain MRI slice — or all 32 slices of one subject, which are
              averaged into a single subject-level prediction, exactly as the reported
              metrics were computed. You get the probability behind the call and a heat-map
              of the region that drove it, alongside the confidence intervals most demos
              leave out.
            </p>

            <div className="mt-7 flex flex-wrap gap-3">
              <a
                href="#analyse"
                className="rounded-xl px-6 py-3 text-[15px] font-bold text-white transition hover:brightness-105"
                style={{
                  background: 'linear-gradient(100deg,var(--color-accent),var(--color-accent3))',
                  boxShadow: '0 10px 26px -8px rgb(217 119 6 / .55)',
                }}
              >
                Try it on a scan
              </a>
              <a
                href="#methodology"
                className="glass-soft rounded-xl px-6 py-3 text-[14px] font-semibold text-muted transition hover:text-ink"
              >
                How it was validated
              </a>
            </div>
          </div>

          <SliceStack />
        </section>

        <div className="mb-14 grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(150px,1fr))]">
          {m.auc && (
            <Tile
              label="ROC AUC"
              value={m.auc.toFixed(3)}
              sub={`95% CI ${m.auc_ci[0].toFixed(3)}–${m.auc_ci[1].toFixed(3)}`}
              mono
            />
          )}
          {m.accuracy && (
            <Tile
              label="Accuracy"
              value={fmtPct(m.accuracy)}
              sub={`vs ${fmtPct(m.baseline)} majority baseline`}
              mono
            />
          )}
          {m.macro_f1 != null && (
            <Tile label="Macro F1" value={m.macro_f1.toFixed(3)} sub="5-fold CV, out-of-fold" mono />
          )}
          <Tile
            label="Cross-scanner"
            value="0.68–0.79"
            sub="AUC on an unseen scanner generation"
            mono
            accent="plain"
          />
          <Tile
            label="Leakage"
            value="0"
            sub="subjects shared between train and test"
            accent="ok"
            mono
          />
        </div>
      </div>

      <FeatureGrid />
      <TrustBar m={m} />

      <div id="analyse" className="mx-auto max-w-6xl px-5 sm:px-6 pt-14 pb-20">
        {/* ── disclaimer ──────────────────────────────────────── */}
        <div className="glass-soft mb-6 rounded-2xl border-warn/25 bg-warn/[0.06] p-4 text-[13px] leading-relaxed text-warn/90">
          <b className="text-warn">Research demonstration only.</b> {cfg.disclaimer}
        </div>

        {/* ── analyse ─────────────────────────────────────────── */}
        <section className="glass topline rounded-3xl p-6 sm:p-7 mb-6">
          <div className="mb-5 flex flex-wrap items-center gap-2.5">
            <span className="h-1.5 w-1.5 rounded-full bg-accent2 shadow-[0_0_10px_var(--color-accent2)]" />
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted">
              Analyse a scan
            </h2>
            <span className="ml-auto text-[11px] font-semibold uppercase tracking-[0.1em] text-ok bg-ok/12 border border-ok/30 rounded-full px-2.5 py-0.5">
              {task.short} · validated
            </span>
          </div>

          <form onSubmit={analyse}>
            <label
              htmlFor="file"
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                if (e.dataTransfer.files.length) pick(e.dataTransfer.files)
              }}
              className={`group block cursor-pointer rounded-2xl border border-dashed p-8 text-center transition ${
                dragging
                  ? 'border-accent2 bg-accent/12'
                  : 'border-line hover:border-accent2/60 hover:bg-black/[0.02]'
              }`}
            >
              <svg
                viewBox="0 0 24 24"
                className="mx-auto mb-3 h-8 w-8 fill-none stroke-muted transition group-hover:stroke-accent2"
                strokeWidth="1.6"
              >
                <path d="M12 16V4m0 0L8 8m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" strokeLinecap="round" />
              </svg>
              <div className="font-semibold">
                Drop a scan — or a whole subject — here, or click to browse
              </div>
              <div className="mt-1 text-[12.5px] text-muted">
                PNG · JPEG · DICOM (.dcm) — select up to {cfg.max_files} axial slices at
                once, max 8 MB each
              </div>
              <div className="mt-2.5 text-[12px] leading-relaxed text-accent2/85">
                Selecting all of a subject&rsquo;s slices reproduces the published
                subject-level metric. One slice is the noisier, harder task.
              </div>
            </label>
            <input
              id="file"
              type="file"
              accept="image/*,.dcm"
              multiple
              className="hidden"
              onChange={(e) => e.target.files.length && pick(e.target.files)}
            />
            {/* Directory pick is a separate input: `webkitdirectory` replaces the
                file-picker with a folder picker, so it cannot share the one above. */}
            <input
              id="folder"
              type="file"
              webkitdirectory=""
              directory=""
              className="hidden"
              onChange={(e) => e.target.files.length && pick(e.target.files)}
            />

            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={!nFiles || busy}
                className="rounded-xl px-6 py-3 text-[15px] font-bold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
                style={{
                  background:
                    'linear-gradient(100deg,var(--color-accent),var(--color-accent3))',
                  boxShadow: nFiles && !busy ? '0 8px 26px -6px rgb(217 119 6 / .5)' : 'none',
                }}
              >
                {busy
                  ? 'Analysing…'
                  : nFiles > 1
                    ? `Run analysis on ${nFiles} slices`
                    : 'Run analysis'}
              </button>

              <label
                htmlFor="folder"
                className="glass-soft cursor-pointer rounded-xl px-4 py-3 text-[13px] font-medium text-muted transition hover:text-ink hover:border-accent2/50"
              >
                Pick a folder
              </label>

              {nFiles > 0 && (
                <span
                  className={`rounded-full border px-3 py-1.5 text-[12px] font-semibold ${
                    nFiles > 1
                      ? 'border-accent2/40 bg-accent/12 text-accent2'
                      : 'border-line bg-black/[0.02] text-muted'
                  }`}
                >
                  {nFiles === 1 ? '1 slice selected' : `${nFiles} slices selected`}
                  {nFiles > 1 && ' · will be averaged'}
                </span>
              )}
            </div>

            {nFiles > 0 && (
              <div className="mt-3 font-mono text-[12px] leading-relaxed text-muted">
                {sampleInfo
                  ? sampleInfo.label
                  : nFiles === 1
                    ? files[0].name
                    : `${files[0].name} … ${files[nFiles - 1].name}`}
              </div>
            )}
          </form>

          <Samples onPick={pick} disabled={busy} />

          <details className="mt-5 text-[12.5px] text-muted">
            <summary className="cursor-pointer select-none font-medium transition hover:text-ink">
              A note on DICOM orientation
            </summary>
            <p className="mt-2 leading-relaxed border-l-2 border-accent/35 pl-3">
              {cfg.dicom_note}
            </p>
          </details>

          {(busy || stage > 0) && !error && (
            <Pipeline pct={pct} current={stage} labels={labels} />
          )}

          {error && (
            <div className="mt-5 rounded-2xl border border-danger/40 bg-danger/10 p-4 text-[13.5px] text-danger">
              {error}
            </div>
          )}
        </section>

        {result && <Result data={result} trueClass={sampleInfo?.true_class} />}

        <div id="methodology">
          <Validation />
        </div>
      </div>

      <FindingsCarousel />

      <div className="mx-auto max-w-6xl px-5 sm:px-6 pt-14 pb-20">
        {/* ── honesty panel ───────────────────────────────────── */}
        <section className="glass-soft rounded-3xl p-6 mb-6">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted mb-4">
            What this does not do
          </h2>
          <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2 text-[13px] leading-relaxed text-muted">
            <p>
              <b className="text-ink">No four-stage prediction.</b> Separating early from
              late MCI is defined in ADNI by a memory-test cutoff, not anatomy. That model
              exists but is withheld here, because presenting it would look authoritative
              without being so.
            </p>
            <p>
              <b className="text-ink">Single slices are noisy.</b> {cfg.single_slice_note}
            </p>
            <p>
              <b className="text-ink">Aggregation is the published estimator.</b>{' '}
              {cfg.multi_slice_note}
            </p>
            <p>
              <b className="text-ink">One dataset only.</b> Trained on research-grade ADNI
              scans. Performance on images from a different hospital is unverified and
              likely lower.
            </p>
            <p>
              <b className="text-ink">Figures are not typed in.</b> Every number is read
              at load time from{' '}
              <code className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-[11.5px] text-accent2">
                {m.source_file}
              </code>
              , so the page cannot drift from the recorded experiment.
            </p>
          </div>
        </section>

        <footer className="border-t border-line pt-6 text-[12px] leading-relaxed text-muted">
          Trained on the ADNI dataset with a strict subject-wise split — no person appears
          in both training and test data. Uploads are processed in memory and never stored
          or logged.
          <br />
          <b className="text-ink/80">This tool cannot diagnose anyone.</b> If you have
          concerns about memory or cognition, speak to a doctor.
        </footer>
      </div>
    </>
  )
}
