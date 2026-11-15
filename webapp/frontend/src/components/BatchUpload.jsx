import { useEffect, useRef, useState } from 'react'
import { batchCsv, groupPatientFiles, patientWarnings, readPrediction } from '../batch'
import Result from './Result'
import PatientResearch from './PatientResearch'
import ResearchLab from './ResearchLab'
import { parseClinicalCsv } from '../clinical'

export default function BatchUpload({ apiBase, taskId, maxFiles, enableResearch = true, onBusyChange }) {
  const [patients, setPatients] = useState([])
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [ignored, setIgnored] = useState(0)
  const [selected, setSelected] = useState(null)
  const [compare, setCompare] = useState(false)
  const [clinicalNote, setClinicalNote] = useState('')
  const controller = useRef(null)
  const stopped = useRef(false)
  const running = useRef(false)
  useEffect(() => () => { stopped.current = true; controller.current?.abort() }, [])
  useEffect(() => { onBusyChange?.(busy || loading) }, [busy, loading, onBusyChange])

  function choose(event) {
    if (running.current) return
    setError('')
    setSelected(null)
    setPatients([])
    setClinicalNote('')
    try {
      const grouped = groupPatientFiles(event.target.files, maxFiles)
      setPatients(grouped.patients)
      setIgnored(grouped.ignored)
    } catch (e) { setError(e.message); setIgnored(0) }
    event.target.value = ''
  }

  async function samples() {
    if (running.current) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch('/samples/index.json')
      if (!response.ok) throw new Error('Could not load sample index.')
      const index = await response.json()
      const subjects = index.filter(s => s.multi).slice(0, 2)
      if (subjects.length !== 2) throw new Error('Two complete sample subjects are unavailable.')
      const rows = await Promise.all(subjects.map(async (s, i) => ({
        id: `Example ${i + 1}`,
        reference: s.true_class,
        status: 'Queued',
        clinical: null,
        files: await Promise.all(s.files.map(async path => {
          const r = await fetch(path)
          if (!r.ok) throw new Error('A sample image could not be loaded.')
          return new File([await r.blob()], path.split('/').pop(), { type: 'image/png' })
        })),
      })))
      setPatients(rows)
      setClinicalNote('Real MRI examples with known ADNI reference groups. Patient demographics are not supplied.')
      setSelected(null)
      setIgnored(0)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  async function clinicalUpload(event) {
    const file = event.target.files[0]
    event.target.value = ''
    if (!file) return
    setLoading(true)
    try {
      if (file.size > 1024 * 1024) throw new Error('Clinical CSV must be smaller than 1 MB.')
      const records = parseClinicalCsv(await file.text())
      const ids = new Set(patients.map(p => p.id))
      const unmatched = [...records.keys()].filter(id => !ids.has(id))
      if (unmatched.length) throw new Error(`${unmatched.length} clinical IDs have no matching patient folder. Correct the CSV before importing.`)
      setPatients(rows => rows.map(p => ({ ...p, clinical: records.get(p.id) || null })))
      setClinicalNote(`${records.size} patient records attached. Clinical values are supplied context and do not affect image predictions.`)
      setError('')
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  async function run() {
    if (running.current) return
    running.current = true
    stopped.current = false
    setBusy(true)
    setError('')
    const update = (id, changes) => setPatients(rows => rows.map(p => p.id === id ? { ...p, ...changes } : p))
    try {
      for (const patient of patients.filter(p => p.status === 'Queued' || p.status === 'Stopped')) {
        if (stopped.current) break
        controller.current = new AbortController()
        update(patient.id, { status: 'Running', progress: 0, error: null })
        try {
          const body = new FormData()
          for (const file of patient.files) body.append('image', file, file.name)
          body.append('task', taskId)
          body.append('research_compare', String(compare))
          const response = await fetch(`${apiBase}/api/predict-stream`, {
            method: 'POST', body, signal: controller.current.signal,
          })
          const result = await readPrediction(response, progress => update(patient.id, { progress }))
          if (stopped.current) { update(patient.id, { status: 'Stopped' }); break }
          update(patient.id, { status: 'Complete', result, progress: 100 })
        } catch (e) {
          update(patient.id, stopped.current
            ? { status: 'Stopped', error: null }
            : { status: 'Failed', error: e.message })
        }
      }
    } finally {
      controller.current = null
      running.current = false
      setBusy(false)
    }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([batchCsv(patients)], { type: 'text/csv;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'patient-batch-results.csv'
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const detail = patients.find(p => p.id === selected)
  const completed = patients.filter(p => p.status === 'Complete').length
  const hasClinical = patients.some(p => p.clinical)
  const pending = patients.some(p => p.status === 'Queued' || p.status === 'Stopped')
  const button = 'rounded-xl border border-line px-4 py-2.5 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed'

  return (
    <section id="batch" className="min-w-0 mb-8">
      <div className="glass topline rounded-3xl p-5 sm:p-7">
        <h2 className="text-2xl font-bold">Analyse a patient batch</h2>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          Choose a folder with one subfolder per person, or try the prepared MRI examples.
          Each person receives one result from their accepted slices.
        </p>
        <details className="mt-3 text-xs leading-relaxed text-muted"><summary className="cursor-pointer">Folder format and upload limits</summary>
          <p className="mt-2">Use Batch / Patient-001 / slices. Include only one scan per person and use codes rather than names.
          Up to 100 people, {maxFiles} images and 48 MB per person, 8 MB per image. PNG, JPEG and supported single-frame axial DICOM.
          Raw sagittal scans need preprocessing. Prefer the complete 32-slice set.</p>
        </details>
        <div className="mt-5 flex flex-wrap gap-3 items-center">
          <label className="text-sm font-semibold" htmlFor="batch-folder">Patient batch folder</label>
          <input id="batch-folder" type="file" webkitdirectory="" directory="" multiple
            disabled={busy || loading} onChange={choose}
            className="min-w-0 max-w-full text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-accent/15 file:px-3 file:py-2" />
          <button type="button" className={button} disabled={busy || loading} onClick={samples}>
            {loading ? 'Loading samples…' : 'Load MRI examples'}
          </button>
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <details className="w-full rounded-xl border border-line p-3">
            <summary className="cursor-pointer text-sm font-semibold">Attach patient details (optional CSV)</summary>
            <div className="mt-3"><label htmlFor="clinical-csv" className="text-sm font-semibold">Patient details CSV</label>
            <input id="clinical-csv" type="file" accept=".csv,text/csv" disabled={busy || loading || !patients.length} onChange={clinicalUpload} className="ml-3 max-w-full text-sm" />
            <p className="mt-2 text-xs text-muted">Columns: patient_id, age, sex, education_years, mmse, moca. Match patient_id exactly to the patient folder; leave unknown values blank. The CSV stays in this browser session and is used for display and export. Clinical fusion has not been evaluated.</p>
            </div>
          </details>
          {clinicalNote && <p role="status" className="w-full text-xs text-muted">{clinicalNote}</p>}
          {enableResearch && <details className="w-full text-sm"><summary className="cursor-pointer text-muted">Advanced research comparison (optional)</summary><label className="mt-3 flex w-full items-center gap-2 text-sm"><input type="checkbox" checked={compare} disabled={busy} onChange={e=>setCompare(e.target.checked)} />Compare experimental attention and GAN-trained models (requires 32 accepted slices per patient)</label></details>}
          <button type="button" className={`${button} bg-accent text-white`} disabled={busy || loading || !pending} onClick={run}>
            {busy ? 'Processing batch…' : 'Analyse batch'}
          </button>
          {busy && <button type="button" className={button} onClick={() => {
            stopped.current = true; controller.current?.abort()
          }}>Stop batch</button>}
          <button type="button" className={button} disabled={busy || !patients.length} onClick={download}>Download CSV</button>
        </div>
        <p role="status" aria-live="polite" className="mt-3 text-sm text-muted">
          {patients.length ? `${completed} of ${patients.length} patients completed. ` : ''}
          {ignored > 0 ? `${ignored} unsupported files excluded. ` : ''}
          {busy ? 'Patients are processed one at a time; results appear as each finishes.' : ''}
        </p>
        {error && <p role="alert" className="mt-3 text-sm text-danger">{error}</p>}
        {compare && completed > 0 && <p className="mt-3 text-sm">Experimental comparisons: {patients.filter(p=>p.result?.research?.available).length} available · {patients.filter(p=>p.result?.research?.disagreement).length} with differing research-model classes. Open “Compare & explain” to inspect each result.</p>}
        {patients.length > 0 && (
          <div className="mt-5 overflow-x-auto rounded-xl border border-line" tabIndex={0} aria-label="Patient batch results">
            <table className="w-full min-w-[760px] text-left text-sm">
              <caption className="p-3 text-left text-xs text-muted">Research predictions, not diagnoses. AD means Alzheimer’s disease; CN means cognitively normal. Probabilities are model outputs, not clinical risk estimates.</caption>
              <thead className="bg-accent/10"><tr>
                {['Person', ...(hasClinical ? ['Supplied patient details'] : []), 'Progress', 'Images used / selected', 'MRI result', 'AD score', 'Reference', ...(compare ? ['Attention AD score', 'GAN-trained AD score'] : []), 'Details'].map(label =>
                  <th key={label} scope="col" className="p-3 font-semibold">{label}</th>)}
              </tr></thead>
              <tbody>{patients.map(p => {
                const r = p.result
                const prob = code => r ? `${((r.ranked.find(c => c.code === code)?.prob ?? 0) * 100).toFixed(1)}%` : '—'
                return <tr key={p.id} className="border-t border-line align-top">
                  <th scope="row" className="max-w-48 break-words p-3 font-semibold">{p.id}</th>
                  {hasClinical && <td className="p-3 text-xs min-w-44">{p.clinical?.demo && <strong className="block mb-1">Fictional demo details</strong>}{p.clinical ? `Age: ${p.clinical.age ?? '—'}; sex: ${p.clinical.sex || '—'}; education: ${p.clinical.education_years ?? '—'} years; MMSE: ${p.clinical.mmse ?? '—'}; MoCA: ${p.clinical.moca ?? '—'}` : 'Not supplied'}</td>}
                  <td className="p-3">{p.status}{p.status === 'Running' ? ` ${p.progress || 0}%` : ''}</td>
                  <td className="p-3 font-mono">{r?.n_slices_used ?? '—'} / {p.files.length}</td>
                  <td className="p-3">{r?.prediction.code || '—'}</td>
                  <td className="p-3 font-mono">{prob('AD')}</td>
                  <td className="p-3">{p.reference ? `${p.reference}${r ? (r.prediction.code === p.reference ? ' · correct' : ' · incorrect') : ''}` : 'Not supplied'}</td>

                  {compare && <td className="p-3">{r?.research?.available ? `${(r.research.arms.attention.ad_probability*100).toFixed(1)}%` : '—'}</td>}
                  {compare && <td className="p-3">{r?.research?.available ? `${(r.research.arms.attention_gan.ad_probability*100).toFixed(1)}%` : '—'}</td>}
                  <td className="p-3"><button type="button" className={button} disabled={!r}
                    onClick={() => setSelected(p.id)} aria-label={`View explanation for ${p.id}`}>View images</button>
                    <details className="mt-2 text-xs text-muted"><summary className="cursor-pointer">{p.error ? 'Check input' : 'Notes'}</summary><p className="mt-2 max-w-56">{patientWarnings(p) || 'Ready for image analysis.'}</p></details>
                  </td>
                </tr>
              })}</tbody>
            </table>
          </div>
        )}
        <p className="mt-4 text-xs leading-relaxed text-muted">AD = Alzheimer’s disease group; CN = cognitively normal group.
          Export the CSV to keep your results. Supplied patient details do not change the MRI prediction.</p>
      </div>
      {detail?.result && <div className="mt-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h3 className="break-all text-lg font-bold">Explanation: {detail.id}</h3>
          <button type="button" className={button} onClick={() => setSelected(null)}>Close explanation</button>
        </div>
        {enableResearch && compare && <PatientResearch result={detail.result} />}
        <Result data={detail.result} trueClass={detail.reference} showResearch={enableResearch && compare} />
      </div>}
      {enableResearch && <details className="mt-5 rounded-xl border border-line p-4"><summary className="cursor-pointer font-semibold">Advanced: experimental comparison</summary><ResearchLab /></details>}
    </section>
  )
}
