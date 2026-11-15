const ROWS = [
  {
    n: '01',
    problem: 'Subject leakage',
    found:
      'Slices from one person appearing in both training and test data inflates accuracy ' +
      'by 36.9 points — measured directly on this dataset, not quoted from a paper.',
    status: 'Avoided by splitting per person before any image is created',
    fixed: true,
  },
  {
    n: '02',
    problem: 'Scanner confound',
    found:
      'Every healthy and Alzheimer’s subject came from one scanner era and every MCI ' +
      'subject from another, so “diagnosis” and “scanner” were the same variable. Models ' +
      'scored 98–100% identifying the scanner while performing at chance on the disease.',
    status:
      'AD and CN span both eras; EMCI and LMCI remain GO2-only. Cross-era testing does not exclude all scanner or site effects.',
    fixed: false,
  },
  {
    n: '03',
    problem: 'Slice misalignment',
    found:
      'Slices were cut at a fixed fraction of image height, landing on different anatomy ' +
      'per subject. Atlas-based coverage checks are still needed to establish which structures were included.',
    status: 'Re-extracted anchored in millimetres below the top of the skull',
    fixed: true,
  },
  {
    n: '04',
    problem: 'Acquisition geometry',
    found:
      'Slices were stacked without the true inter-slice spacing, so every scan was ' +
      'stretched by a class-correlated amount that tracked the scanning protocol.',
    status: 'Isotropic v4 preprocessing is available, but the deployed checkpoint was trained on v3. The geometry correction has not been validated in the deployed model.',
    fixed: false,
  },
]

/**
 * The methodology section. For a technical reader this is the substance: the accuracy
 * number matters far less than whether it means anything.
 */
export default function Validation({ crossEraRange }) {
  return (
    <section className="glass topline rounded-3xl p-6 sm:p-7 mb-6">
      <div className="flex items-center gap-2.5 mb-2">
        <span className="h-1.5 w-1.5 rounded-full bg-cyan shadow-[0_0_10px_var(--color-cyan)]" />
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted">
          Methodology
        </h2>
      </div>
      <h3 className="text-2xl font-bold tracking-tight mb-2">
        Four sources of <span className="grad-accent">evaluation bias</span>
      </h3>
      <p className="text-sm text-muted max-w-2xl mb-6">
        These checks examine sources of bias in this dataset. The status below
        distinguishes implemented corrections from limitations that remain.
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {ROWS.map((r) => (
          <article key={r.n} className="glass-soft rounded-2xl p-4 flex flex-col">
            <div className="flex items-center justify-between mb-2.5">
              <span className="font-mono text-[11px] text-accent2">{r.n}</span>
              {r.fixed && (
                <span className="text-[9.5px] font-bold uppercase tracking-[0.1em] text-ok bg-ok/12 border border-ok/30 rounded-full px-2 py-0.5">
                  resolved
                </span>
              )}
            </div>
            <h4 className="font-semibold mb-2">{r.problem}</h4>
            <p className="text-[12.5px] leading-relaxed text-muted mb-3 grow">{r.found}</p>
            <p className="text-[12.5px] leading-relaxed text-ink/85 border-t border-line pt-2.5">
              {r.status}
            </p>
          </article>
        ))}
      </div>

      <div className="glass-soft rounded-2xl p-4 mt-4 flex flex-wrap gap-x-8 gap-y-3 items-center">
        <div>
          <div className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-muted">
            Cross-era transfer experiment
          </div>
          <div className="text-sm mt-1">
            Trained on one scanner generation, tested on a completely different one
          </div>
        </div>
        <div className="font-mono text-2xl font-bold grad-accent tabular-nums">
          {crossEraRange ? `AUC ${crossEraRange.map(x => x.toFixed(3)).join('–')}` : 'Recorded research runs'}
        </div>
        <p className="text-[12px] leading-relaxed text-muted basis-full">
          These experiments test transfer between acquisition eras without shared
          subjects. They provide evidence of transfer, but do not establish that
          the model is independent of scanner, site or acquisition effects.
        </p>
      </div>
    </section>
  )
}
