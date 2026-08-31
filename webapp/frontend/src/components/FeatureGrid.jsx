const FEATURES = [
  {
    title: '853 Real Clinical Scans',
    body: 'Raw DICOM downloaded directly from the ADNI study archive — not a pre-packaged teaching dataset.',
  },
  {
    title: 'Subject-Level Splits',
    body: 'No person’s slices appear in both training and test data. Unfixed, that leak alone inflates accuracy by 36.9 points.',
  },
  {
    title: '5-Fold Cross-Validated',
    body: 'The headline number is measured on every subject, not a favourable single split — three earlier headlines were withdrawn this way.',
  },
  {
    title: '4 Confounds Found & Fixed',
    body: 'Scanner era, slice misalignment, and acquisition geometry were each measured as shortcuts and corrected before being trusted.',
  },
  {
    title: 'Grad-CAM Explainability',
    body: 'Every prediction ships a heat-map of the region that drove it — the same check that first caught the scanner-artifact shortcut.',
  },
  {
    title: 'Cross-Scanner Tested',
    body: 'Trained on one scanner generation, evaluated on a completely different one: the signal survived in 5 of 6 runs.',
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
            Why this result is trustworthy
          </div>
          <h2 className="text-[clamp(1.6rem,3.4vw,2.2rem)] font-black tracking-tight">
            Built to survive scrutiny, not just a demo click
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
