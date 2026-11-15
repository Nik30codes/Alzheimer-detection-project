import { useEffect, useState } from 'react'
import ConfusionMatrix from './ConfusionMatrix'
const pct = value => `${(value * 100).toFixed(1)}%`
const names = { demographics: 'Age, sex and education', biomarkers: 'Blood measurements', 'clinical measurements': 'Clinical assessments', 'MRI + tabular fusion': 'MRI + clinical information' }

export default function BenchmarkEvidence({ apiBase }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    fetch(`${apiBase}/api/research/benchmark`, { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error('Recorded results are unavailable. Please refresh and try again.'); return r.json() })
      .then(setData).catch(e => { if (e.name !== 'AbortError') setError(e.message) })
    return () => controller.abort()
  }, [apiBase])
  const m = data?.mri.metrics
  return <section id="benchmark" className="glass topline rounded-3xl p-5 sm:p-7 mb-6 scroll-mt-6">
    <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted">RECORDED STUDY RESULTS</p>
    <h2 className="mt-2 text-2xl font-bold">Evaluation and clinical benchmark</h2>
    <p className="mt-3 text-sm leading-relaxed text-muted">Explore the saved MRI results and the data prepared for the next comparison. Upload results above are separate from this evaluation.</p>
    {error && <p role="alert" className="mt-4 text-danger">{error}</p>}
    {!data && !error && <p role="status" className="mt-4 text-sm text-muted">Loading recorded results…</p>}
    {data && <div className="mt-5 divide-y divide-line">
      <details className="py-4"><summary className="cursor-pointer font-semibold">MRI evaluation: correct predictions, mistakes and uncertainty</summary>
        <div className="mt-4 grid gap-6 md:grid-cols-[1fr_auto]"><div className="text-sm leading-relaxed text-muted">
          <p><strong className="text-ink">{m.confusion_matrix[0][0] + m.confusion_matrix[1][1]} of {m.subjects} participants</strong> were classified correctly ({pct(m.accuracy)}). Accuracy interval: {m.accuracy_wilson_95ci.map(pct).join('–')}.</p>
          <p className="mt-3">The model missed {m.confusion_matrix[1][0]} AD cases and classified {m.confusion_matrix[0][1]} cognitively normal participants as AD. These internal results do not establish clinical performance at a new hospital.</p>
          <p className="mt-3 text-xs">AD score is a model output, not medical risk. Each fold used a threshold chosen on validation participants.</p></div><ConfusionMatrix matrix={m.confusion_matrix} classes={m.classes} /></div>
        <h3 className="mt-5 text-sm font-semibold">Examples from the saved evaluation</h3>
        <div className="mt-3 overflow-x-auto rounded-xl border border-line"><table className="w-full min-w-[520px] text-left text-sm"><caption className="p-3 text-left text-xs text-muted">First stored example of each outcome, including mistakes. No participant identifiers are shown.</caption>
          <thead className="bg-accent/10"><tr>{['Example','Reference','Model result','AD score','Outcome'].map(t => <th className="p-3" key={t}>{t}</th>)}</tr></thead>
          <tbody>{data.recorded_examples?.map(e => <tr className="border-t border-line" key={e.id}><th scope="row" className="p-3">{e.id}</th><td className="p-3">{e.reference}</td><td className="p-3">{e.prediction}</td><td className="p-3 font-mono">{pct(e.ad_score)}</td><td className="p-3">{e.correct ? 'Correct' : 'Incorrect'}</td></tr>)}</tbody></table></div>
        <a href="/samples/benchmark-examples.csv" download className="mt-3 inline-block text-sm font-semibold text-accent underline underline-offset-4">Download recorded examples</a>
      </details>
      <details className="py-4"><summary className="cursor-pointer font-semibold">Clinical data and the next benchmark</summary>
        <p className="mt-3 text-sm leading-relaxed text-muted">Diagnosis, demographics, cognitive assessments and blood-test exports have been audited. Exact MRI-visit matching and official image-quality review remain incomplete. Attached clinical details are displayed as supplied; the current model uses MRI only.</p>
        <div className="mt-4 overflow-x-auto rounded-xl border border-line"><table className="w-full min-w-[520px] text-left text-sm"><caption className="p-3 text-left text-xs text-muted">Clinical and combined-model accuracy remains unmeasured.</caption><thead className="bg-accent/10"><tr>{['Information used','Current stage','Accuracy'].map(t => <th key={t} className="p-3">{t}</th>)}</tr></thead><tbody>
          <tr className="border-t border-line"><th scope="row" className="p-3">MRI images</th><td className="p-3">Internal evaluation complete</td><td className="p-3 font-mono">{pct(m.accuracy)}</td></tr>
          {data.arms.map(a => <tr key={a.name} className="border-t border-line"><th scope="row" className="p-3">{names[a.name] || a.name}</th><td className="p-3">Cohort and visit review</td><td className="p-3">Not tested</td></tr>)}</tbody></table></div>
        {data.candidate_linkage && <p className="mt-3 text-sm text-muted">{data.candidate_linkage.summaries.ad_cn.unique_candidates_within_30d} AD/CN participants have a candidate assessment within 30 days before or on the MRI date. These are candidates, not a verified matched cohort.</p>}
        <p className="mt-3 text-sm text-muted">The planned comparison uses the same eligible people and participant-separated folds for MRI-only, clinical-only and combined models.</p>
      </details>
      <details className="py-4"><summary className="cursor-pointer font-semibold">Data checks and technical results</summary>
        {data.input_audit && <p className="mt-3 text-sm leading-relaxed text-muted">Checked {data.input_audit.mri_headers.files.toLocaleString()} selected DICOM headers and {data.input_audit.processed_images.referenced_files.toLocaleString()} prepared images. One participant-ID mismatch remains unresolved. File checks do not replace clinical image review.</p>}
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">{[[pct(m.sensitivity),'Sensitivity'],[pct(m.specificity),'Specificity'],[pct(m.ppv),'Precision'],[pct(m.npv),'Negative predictive value'],[m.auprc.toFixed(4),'AUPRC'],[m.brier_score.toFixed(4),'Brier score']].map(([v,label]) => <div key={label}><dt className="text-xs text-muted">{label}</dt><dd className="mt-1 font-mono font-semibold">{v}</dd></div>)}</dl>
        <p className="mt-4 text-xs leading-relaxed text-muted">An earlier draft reported 74.1% after selecting a threshold on pooled out-of-fold scores. The website uses stored fold classifications: 70.9% accuracy and ROC AUC 0.7845.</p>
        {data.assay_audit && <details className="mt-4 text-sm"><summary className="cursor-pointer">Blood-test availability</summary><p className="mt-2 text-xs text-muted">Whole-export counts; matched MRI availability may be smaller. Missing codes remain missing, and assay platforms are kept separate.</p><ul className="mt-2 space-y-1 text-xs text-muted">{data.assay_audit.analytes.map(a => <li key={a.field}>{a.field}: {a.measured_rows} measured, {a.missing_rows} missing of {data.assay_audit.rows}</li>)}</ul></details>}
        <details className="mt-4 text-sm"><summary className="cursor-pointer">Requirements before the combined comparison</summary><ul className="mt-2 list-disc pl-5 space-y-1 text-xs text-muted">{data.protocol.release_gates.map(g => <li key={g}>{g}</li>)}</ul></details>
      </details>
    </div>}
  </section>
}
