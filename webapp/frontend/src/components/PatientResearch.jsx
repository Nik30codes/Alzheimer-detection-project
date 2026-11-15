import ExplanationGuide from './ExplanationGuide'

export default function PatientResearch({ result }) {
  const r = result.research
  return <section className="glass rounded-2xl p-5 mb-5">
    <h3 className="text-xl font-bold">Experimental comparison for this patient</h3>
    {!r?.available ? <p className="mt-3 text-sm">{r?.reason || 'Comparison was not enabled for this run.'}</p> : <>
      <p className="mt-3 text-sm text-muted">Research fold {r.fold}. {r.note} {r.preprocessing}</p>
      <p className="mt-2 font-semibold">{r.disagreement ? 'The research models disagree. This is a reason for caution, not a diagnosis.' : 'The research models agree on a class; agreement does not establish a diagnosis.'}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">{Object.entries(r.arms).map(([key,m]) => <div key={key} className="rounded-xl border border-line p-3">
        <h4 className="font-semibold">{{mean:'Mean pooling',attention:'Learned attention',attention_gan:'Attention + GAN training'}[key]}</h4>
        <p className="mt-2 text-lg">{m.prediction} · {(m.ad_probability*100).toFixed(1)}% AD score</p>
        <p className="text-xs text-muted">Fold validation threshold: {m.threshold.toFixed(3)}</p>
        {key!=='mean' && <><p className="mt-3 text-sm font-semibold">Top weighted slices</p><ol className="mt-2 text-xs space-y-2">{m.top_slices.map(s => <li key={s.index}><span className="break-all">{s.file}</span> · {(s.weight*100).toFixed(1)}%<div className="mt-1 h-1.5 rounded bg-black/10"><div className="h-1.5 rounded bg-accent" style={{width:`${s.weight*100}%`}} /></div></li>)}</ol></>}
      </div>)}</div>
      <p className="mt-3 text-xs text-muted">Weights sum to 100% across all accepted slices; only the top five are shown. Attention weights are not damaged-tissue maps. GAN images were used during training only; your uploaded scans are unchanged. These models did not outperform the mean-pooling baseline in the recorded experiment.</p>
    </>}
    <ExplanationGuide />
  </section>
}
