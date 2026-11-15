const FEATURES = [
  {
    title: '853 Research Participants',
    body: 'The project uses MRI scans from the approved ADNI research archive.',
  },
  {
    title: 'Subject-Level Splits',
    body: 'All slices from one person stay in the same evaluation partition.',
  },
  {
    title: '5-Fold Cross-Validated',
    body: 'Results use one held-out prediction per person across five folds.',
  },
  {
    title: 'Confounds Investigated',
    body: 'Scanner effects and preprocessing shortcuts were investigated; limitations remain.',
  },
  {
    title: 'Grad-CAM Explainability',
    body: 'Inspect the image regions that influenced a result using Grad-CAM.',
  },
  {
    title: 'Cross-Scanner Tested',
    body: 'Six experiments tested transfer between MRI acquisition eras.',
  },
]

/** Feature-grid section — structurally borrowed from the reference screenshots'
 * 2x3 checkmark card grid, content entirely specific to this project. */
export default function FeatureGrid() {
  return (
    <section className="py-14 sm:py-16">
      <div className="mx-auto max-w-6xl px-5 sm:px-6">
        <div className="max-w-2xl mb-10">
          <div className="text-[11px] font-semibold uppercase tracking-[0.15em] text-accent mb-3">
            Research quality checks
          </div>
          <h2 className="text-[clamp(1.6rem,3.4vw,2.2rem)] font-black tracking-tight">
            How we checked the results
          </h2>
        </div>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f) => (
            <div key={f.title} className="flex gap-3.5">
              <div className="badge-check mt-0.5">
                <svg viewBox="0 0 20 20" className="h-4.5 w-4.5 fill-none stroke-current" strokeWidth="2.4">
                  <path d="M4 10.5l3.6 3.6L16 6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div>
                <h3 className="font-bold text-[15px] mb-1.5">{f.title}</h3>
                <p className="text-[13px] leading-relaxed text-muted">{f.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
