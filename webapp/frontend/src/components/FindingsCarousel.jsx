import { useEffect, useState } from 'react'

/**
 * Rotating strip, structurally in place of the reference screenshots' testimonial
 * carousel — same layout (quote, byline, arrows, progress dots), but there are no real
 * user testimonials for a research demo, and inventing quotes attributed to made-up
 * people would misrepresent the project. Instead it cycles real, citable findings
 * pulled straight from docs/PROJECT_EXPLANATION.md, credited to the report itself.
 */
const FINDINGS = [
  {
    quote:
      'The 95% confidence interval for ROC AUC excludes 0.5 by a wide margin — the result is not chance.',
    from: 'PROJECT_EXPLANATION.md · §2, the result',
  },
  {
    quote:
      'This project found four dataset shortcuts by measurement, not by hoping — and fixed three of them before trusting the headline.',
    from: 'PROJECT_EXPLANATION.md · §4, the four problems found',
  },
  {
    quote:
      'Re-measuring the identical model with 5-fold cross-validation revealed the single-split headline was a favourable draw — caught before it shipped, three separate times.',
    from: 'PROJECT_EXPLANATION.md · §7, the most important lesson',
  },
  {
    quote:
      'Early models put 77% of their Grad-CAM attention outside the brain. The current model concentrates on ventricles and the temporal lobe — the anatomy Alzheimer’s actually damages.',
    from: 'PROJECT_EXPLANATION.md · §5, where the model looks',
  },
]

export default function FindingsCarousel() {
  const [i, setI] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setI((v) => (v + 1) % FINDINGS.length), 7000)
    return () => clearInterval(id)
  }, [])

  const go = (d) => setI((v) => (v + d + FINDINGS.length) % FINDINGS.length)
  const f = FINDINGS[i]

  return (
    <div className="band-tint">
      <div className="mx-auto max-w-3xl px-5 sm:px-6 py-14 sm:py-16 text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent mb-6">
          What the measurements actually showed
        </div>
        <div className="flex items-center justify-center gap-4 sm:gap-6">
          <button
            type="button"
            aria-label="Previous finding"
            onClick={() => go(-1)}
            className="glass grid h-9 w-9 shrink-0 place-items-center rounded-full text-muted transition hover:text-accent"
          >
            ←
          </button>
          <p key={i} className="fade-slide min-h-[104px] text-[19px] sm:text-[21px] font-semibold leading-snug">
            “{f.quote}”
          </p>
          <button
            type="button"
            aria-label="Next finding"
            onClick={() => go(1)}
            className="glass grid h-9 w-9 shrink-0 place-items-center rounded-full text-muted transition hover:text-accent"
          >
            →
          </button>
        </div>
        <div className="mt-4 text-[12.5px] font-medium text-muted">{f.from}</div>
        <div className="mt-6 flex justify-center gap-2">
          {FINDINGS.map((_, d) => (
            <button
              key={d}
              type="button"
              aria-label={`Go to finding ${d + 1}`}
              onClick={() => setI(d)}
              className={`h-1.5 rounded-full transition-all ${
                d === i ? 'w-6 bg-accent' : 'w-1.5 bg-line'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
