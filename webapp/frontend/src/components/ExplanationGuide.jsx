import { useState } from 'react'

const TOPICS = [
  ['probability', 'Understand probabilities'], ['heatmap', 'Read the heatmap'],
  ['attention', 'What is learned attention?'], ['augmentation', 'Synthetic augmentation'],
  ['evaluation', 'Understand evaluation'], ['batch', 'Patient grouping'], ['privacy', 'Data privacy'],
]

export default function ExplanationGuide() {
  const [answer, setAnswer] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [topic, setTopic] = useState('')
  async function ask(id) {
    setBusy(true); setError(''); setAnswer(null); setTopic(id)
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE || ''}/api/explain/${id}`)
      if (!response.ok) throw new Error('Explanation service unavailable. Please try again.')
      setAnswer(await response.json())
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  return <aside className="mt-6 rounded-2xl border border-line p-4">
    <h3 className="font-semibold">Understand this research result</h3>
    <p className="mt-2 text-xs leading-relaxed text-muted">Local source-backed explanations. No images or patient identifiers are passed to the language model. This guide explains the project; it does not interpret your health.</p>
    <div className="mt-3 flex flex-wrap gap-2">{TOPICS.map(([id, label]) =>
      <button key={id} type="button" disabled={busy} onClick={() => ask(id)}
        className="rounded-lg border border-line px-3 py-2 text-xs disabled:opacity-50">{label}</button>)}</div>
    <div role="status" aria-live="polite" className="mt-3 text-sm text-muted">
      {busy && 'Retrieving sources and checking the local model. The first request can take longer.'}
      {error}
      {answer && <>
        <h4 className="font-semibold text-ink">{TOPICS.find(([id]) => id === topic)?.[1]}</h4>
        <p className="mt-1 text-xs">{answer.status}</p>
        {answer.sources.map(source => <div key={source.id} className="mt-3 border-l-2 border-accent/40 pl-3">
          <p className="font-medium text-ink">{source.title}</p>
          <p className="mt-1 leading-relaxed">{source.text}</p>
          <p className="mt-1 text-xs">Source: {source.url ? <a href={source.url} target="_blank" rel="noopener noreferrer" className="underline">{source.source}</a> : source.source}</p>
        </div>)}
      </>}
    </div>
  </aside>
}
