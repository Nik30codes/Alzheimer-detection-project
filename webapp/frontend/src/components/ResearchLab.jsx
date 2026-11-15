import { useEffect, useState } from 'react'
import ExplanationGuide from './ExplanationGuide'

const BASE = import.meta.env.VITE_API_BASE || ''
const names = { mean: 'Mean pooling', attention: 'Learned attention', attention_gan: 'Attention + improved DCGAN' }
const pct = value => `${(value * 100).toFixed(2)}%`

export default function ResearchLab() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    fetch(`${BASE}/api/research/results`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Research results could not be loaded.'); return response.json() })
      .then(setData).catch(e => { if (e.name !== 'AbortError') setError(e.message) })
    return () => controller.abort()
  }, [])
  return <section id="research" className="glass-soft rounded-3xl border border-line p-5 sm:p-8 my-6 scroll-mt-6">
    <div className="flex flex-wrap items-center gap-3">
      <span className="rounded-full bg-accent/10 px-3 py-1 text-xs font-semibold text-accent2">Separate research edition</span>
      <span className="text-xs text-muted">Recorded experiments · 26 October 2026</span>
    </div>
    <h2 className="mt-4 text-3xl font-bold tracking-tight">Attention, augmentation & local explanations</h2>
    <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">Explore the measured results of learned slice attention and conditional DCGAN augmentation. Batch uploads keep the existing classifier and optionally run the research models alongside it. The aggregate table below describes the recorded experiment; individual results appear in your patient rows.</p>
    <div className="mt-5 grid gap-3 sm:grid-cols-3">
      {[['Attention','Learns how to weight MRI slice features within a patient.'],['Conditional DCGAN','Generates independent training slices, not complete patient scans.'],['Local LLM + retrieval','Selects cited project explanations locally; it does not diagnose patients.']].map(([title, text]) => <div key={title} className="rounded-xl border border-line p-4"><h3 className="font-semibold">{title}</h3><p className="mt-2 text-sm text-muted">{text}</p></div>)}
    </div>
    {error && <p role="alert" className="mt-4 text-danger">{error}</p>}
    {!data && !error && <p role="status" className="mt-4">Loading recorded research results…</p>}
    {data && <>
      <div className="mt-6 rounded-xl border border-amber-600/30 bg-amber-50 p-4 text-sm text-amber-950">
        <strong>Research finding: improved image quality did not establish improved prediction.</strong>
        <p className="mt-1">The experimental models have not replaced the website classifier. Synthetic image anatomy and disease labels remain unverified.</p>
      </div>
      <h3 className="mt-6 mb-3 text-lg font-semibold">Patient-level comparison</h3>
      <div className="overflow-x-auto rounded-xl border border-line" role="region" aria-label="Experimental model comparison" tabIndex={0}>
        <table className="w-full min-w-[680px] text-left text-sm">
          <caption className="p-3 text-left text-xs text-muted">501 patients · five folds · AD is the positive class · thresholds selected on inner validation patients</caption>
          <thead><tr className="border-y border-line bg-black/5">{['Research model','AUC (95% interval)','Accuracy','Sensitivity','Specificity'].map(x => <th key={x} className="p-3 font-semibold">{x}</th>)}</tr></thead>
          <tbody>{Object.entries(names).map(([key, label]) => { const m=data.evaluation.arms[key]; return <tr key={key} className="border-b border-line"><th scope="row" className="p-3 font-medium">{label}</th><td className="p-3 whitespace-nowrap">{m.auroc.toFixed(4)} ({m.patient_bootstrap_95ci.auroc.map(x=>x.toFixed(3)).join('–')})</td><td className="p-3">{pct(m.accuracy)}</td><td className="p-3">{pct(m.sensitivity)}</td><td className="p-3">{pct(m.specificity)}</td></tr>})}</tbody>
        </table>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-muted">These are exploratory results from a previously studied cohort, not external clinical validation. Confidence intervals resample patients conditional on the fitted folds. This frozen-encoder experiment differs from the classifier serving uploads.</p>
      <h3 className="mt-7 text-lg font-semibold">What changed in synthetic image quality?</h3>
      <p className="mt-2 text-sm text-muted">Training increased from 150 to 3,000 updates. Standard initialization alone still produced nearly identical images; discriminator normalization restored more variation. Cleaner images do not prove disease fidelity.</p>
      <figure className="mt-4 overflow-hidden rounded-xl border border-line bg-white"><img src={`${BASE}/api/research/figures/synthetic_comparison.png`} alt="Fixed samples comparing the noisy original GAN, collapsed longer training, and more varied normalized GAN for CN and AD conditioning" className="w-full h-auto" loading="lazy"/><figcaption className="p-3 text-xs text-muted">First four fixed samples per class, without selecting attractive examples. Outputs are 64 pixels and are not genuine patient scans.</figcaption></figure>
      <details className="mt-4 rounded-xl border border-line p-4"><summary className="cursor-pointer font-medium">View diversity and image-statistics checks</summary><img src={`${BASE}/api/research/figures/quality_metrics.png`} alt="Diversity relative to real validation images and intensity/texture mismatch across three GAN settings" className="mt-3 w-full h-auto" loading="lazy"/><p className="mt-2 text-xs text-muted">Pixel statistics are technical diagnostics, not anatomical or clinical quality scores. Same-class nearest-image checks do not guarantee absence of memorization.</p></details>
    </>}
    <ExplanationGuide />
    <div className="mt-5 flex flex-wrap gap-4 text-sm"><a href="#batch" className="underline text-accent2">Try patient batch uploads</a><a href="http://127.0.0.1:8765/" target="_blank" rel="noopener noreferrer" className="underline text-accent2">Open original website</a></div>
  </section>
}
