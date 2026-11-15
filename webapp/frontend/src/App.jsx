import { useEffect, useState } from 'react'
import BrainBackground from './components/BrainBackground'
import FeatureGrid from './components/FeatureGrid'
import BatchUpload from './components/BatchUpload'
import BenchmarkEvidence from './components/BenchmarkEvidence'
import { readPrediction } from './batch'
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
  const [mode, setMode] = useState('batch')
  const [batchBusy, setBatchBusy] = useState(false)
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
      .then((r) => { if (!r.ok) throw new Error('The model service is unavailable.'); return r.json() })
      .then(setCfg)
      .catch(() => setError('Could not connect to the model service. Please refresh and try again.'))
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
      const prediction = await readPrediction(res, (progress, event) => {
        setPct(progress)
        if (event?.stage) {
          setLabels(l => ({ ...l, [event.stage]: event.label }))
          setStage(event.done_stage ? event.stage + 1 : event.stage)
        }
      })
      setResult(prediction)
      setPct(100)
      setStage(7)
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
              href="#benchmark"
              className="glass-soft hidden sm:block rounded-full px-4 py-2 text-[12.5px] font-medium text-muted transition hover:text-ink"
            >
              Study results
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
                Evaluated on {m.n_test} participants
              </span>
              <span className="text-[12px] text-ok/70">· ADNI</span>
            </div>

            <h1 className="max-w-xl text-[clamp(2.2rem,5.4vw,3.7rem)] font-black leading-[1.05] tracking-[-0.02em]">
              <span className="grad">Alzheimer&rsquo;s detection</span>
              <br />
              you can actually audit.
            </h1>

            <p className="mt-6 max-w-xl text-[16px] leading-relaxed text-muted">
              Upload one MRI or a batch of patient folders. Review one result per person,
              export the table, and inspect the image regions that influenced the model.
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
                Try an MRI batch
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
            value={m.cross_era_auc_range ? m.cross_era_auc_range.map(x => x.toFixed(2)).join('–') : 'Under review'}
            sub="AUC across six cross-era experiments"
            mono
            accent="plain"
          />
          <Tile
            label="Leakage"
            value="0"
            sub="participants shared across manifest splits"
            accent="ok"
            mono
          />
        </div>
      </div>

      <FeatureGrid />
      <TrustBar />

      <div id="analyse" className="mx-auto max-w-6xl px-5 sm:px-6 pt-14 pb-20">
        {/* ── disclaimer ──────────────────────────────────────── */}
        <div className="glass-soft mb-6 rounded-2xl border-warn/25 bg-warn/[0.06] p-4 text-[13px] leading-relaxed text-warn/90">
          <b className="text-warn">Research demonstration.</b> Model outputs are not diagnoses. Clinical use has not been validated.
        </div>

        {/* ── analyse ─────────────────────────────────────────── */}
        <div className="mb-5 flex flex-wrap gap-2" role="group" aria-label="Analysis mode">
          {[['batch', 'Patient batch'], ['single', 'One person']].map(([id, label]) => <button key={id}
            type="button" aria-pressed={mode === id} disabled={busy || batchBusy}
            onClick={() => setMode(id)} className={`rounded-full border px-5 py-2.5 text-sm font-semibold transition disabled:opacity-40 ${mode === id ? 'border-accent bg-accent text-white' : 'border-line bg-panel text-muted hover:text-ink'}`}>{label}</button>)}
        </div>
        <div hidden={mode !== 'batch'}><BatchUpload apiBase={API_BASE} taskId={task.id} maxFiles={cfg.max_files} enableResearch={false} onBusyChange={setBatchBusy} /></div>
        <div hidden={mode !== 'single'}>
        <section className="glass topline rounded-3xl p-6 sm:p-7 mb-6">
          <div className="mb-5 flex flex-wrap items-center gap-2.5">
            <span className="h-1.5 w-1.5 rounded-full bg-accent2 shadow-[0_0_10px_var(--color-accent2)]" />
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted">
              Analyse a scan
            </h2>
            <span className="ml-auto text-[11px] font-semibold uppercase tracking-[0.1em] text-ok bg-ok/12 border border-ok/30 rounded-full px-2.5 py-0.5">
              {task.short} · research model
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
                Use the complete 32-slice axial set from one person where possible.
              </div>
              <div className="mt-2.5 text-[12px] leading-relaxed text-warn/85">
                Analysis time depends on the server and the number of slices. Progress is shown below.
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
        </div>
        <BenchmarkEvidence apiBase={API_BASE} />
        <div id="methodology" className="scroll-mt-6">
          <details className="glass rounded-3xl p-5 sm:p-7">
            <summary className="cursor-pointer font-bold text-lg">Methodology and remaining limitations</summary>
            <div className="mt-5"><Validation crossEraRange={m.cross_era_auc_range} /></div>
          </details>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-5 sm:px-6 pb-12">
        <footer className="border-t border-line pt-6 text-[12px] leading-relaxed text-muted">
          Nikunj Bhalla · 2430030052 · B.Tech Information Technology · Manipal University Jaipur
          <br />
          ADNI research data. <b className="text-ink/80">Research demonstration, not a diagnosis.</b>
        </footer>
      </div>
    </>
  )
}
