import ConfusionMatrix from './ConfusionMatrix'
import Tile from './Tile'

const pct = (p) => `${(p * 100).toFixed(1)}%`

/** Result panel: verdict banner, metric row, probability bars, input vs Grad-CAM. */
export default function Result({ data, trueClass }) {
  const p = data.prediction
  // When a built-in sample is used the ground truth is known, so state plainly whether
  // the model got it right. Hiding a miss would make the demo dishonest.
  const correct = trueClass ? p.code === trueClass : null
  const isAD = p.code === 'AD'
  const n = data.n_slices_used ?? 1
  const multi = n > 1
  const skipped = data.n_slices_skipped || 0
  const range = data.slice_prob_range

  return (
    <section className="glass topline rounded-3xl p-6 sm:p-7 mb-6 rise">
      <div className="flex flex-wrap items-center gap-2.5 mb-5">
        <span className="h-1.5 w-1.5 rounded-full bg-accent3 shadow-[0_0_10px_var(--color-accent3)]" />
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted">
          Analysis result
        </h2>
        <span
          className={`ml-auto rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.1em] ${
            multi
              ? 'border-accent2/40 bg-accent/12 text-accent2'
              : 'border-warn/30 bg-warn/8 text-warn'
          }`}
        >
          {multi ? `subject-level · ${n} slices` : 'single slice'}
        </span>
      </div>

      {/* The headline claim of this whole feature: the answer is an average, not one
          slice's guess. It goes above the verdict because it changes how the verdict
          should be read. */}
      <div
        className={`mb-5 rounded-2xl border p-4 ${
          multi
            ? 'border-accent2/30 bg-accent/[0.07]'
            : 'border-line bg-black/[0.02]'
        }`}
      >
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span
            className={`text-[19px] font-extrabold leading-none ${
              multi ? 'grad-accent' : 'text-ink'
            }`}
          >
            {multi ? `Averaged over ${n} slices` : 'Based on a single slice'}
          </span>
          {skipped > 0 && (
            <span className="rounded-full border border-warn/30 bg-warn/10 px-2.5 py-0.5 font-mono text-[11.5px] font-semibold text-warn">
              {skipped} skipped
            </span>
          )}
        </div>
        <p className="mt-2 text-[12.5px] leading-relaxed text-muted">
          {data.aggregation}
        </p>
        {data.skipped_note && (
          <p className="mt-2 text-[12.5px] leading-relaxed text-warn/90">
            {data.skipped_note}
          </p>
        )}
        {multi && range && (
          <p className="mt-2 text-[12.5px] leading-relaxed text-muted">
            Individual slices ranged from{' '}
            <span className="font-mono text-accent2">{pct(range[0])}</span> to{' '}
            <span className="font-mono text-accent2">{pct(range[1])}</span>{' '}
            probability of {data.positive_class}
            {data.slice_votes && (
              <>
                {' '}— judged alone they would have split{' '}
                <span className="font-mono">
                  {Object.entries(data.slice_votes)
                    .map(([k, v]) => `${v} ${k}`)
                    .join(' / ')}
                </span>
              </>
            )}
            . That spread is the reason aggregation is the published estimator.
          </p>
        )}
      </div>

      {data.mixed_subjects && (
        <div className="mb-5 rounded-2xl border border-danger/45 bg-danger/10 p-4 text-sm text-danger">
          <b>These slices are from more than one person.</b>
          <p className="mt-1.5 leading-relaxed text-danger/90">
            {data.mixed_subjects_note}
          </p>
        </div>
      )}

      {trueClass && (
        <div
          className={`rounded-2xl px-4 py-3 mb-5 text-sm border ${
            correct
              ? 'border-ok/35 bg-ok/8 text-ok'
              : 'border-warn/35 bg-warn/8 text-warn'
          }`}
        >
          {correct ? (
            <>
              <b>Correct.</b> True diagnosis was {trueClass}; the model predicted {p.code}
              {multi ? ` from ${n} slices averaged together` : ' from a single slice'}.
            </>
          ) : (
            <>
              <b>Incorrect.</b> True diagnosis was {trueClass}; the model predicted{' '}
              {p.code}. Shown rather than hidden — at ~83% subject-level accuracy this
              happens
              {multi ? '.' : ', and a single slice is noisier still.'}
            </>
          )}
        </div>
      )}

      <div className="grid gap-3 mb-6 [grid-template-columns:repeat(auto-fit,minmax(146px,1fr))]">
        <Tile
          label="Prediction"
          value={p.code}
          sub={p.label}
          accent={isAD ? 'warn' : 'ok'}
        />
        <Tile
          label="Confidence"
          value={pct(p.prob)}
          sub={multi ? `mean over ${n} slices` : 'model probability'}
          mono
        />
        <Tile
          label="Slices used"
          value={multi ? `${n}` : '1'}
          sub={
            skipped > 0
              ? `${skipped} skipped as unusable`
              : multi
                ? 'averaged into one prediction'
                : 'no aggregation — noisier'
          }
          mono
          accent={skipped > 0 ? 'warn' : multi ? 'grad' : 'plain'}
        />
        <Tile label="Latency" value={`${data.elapsed}s`} sub="end to end" mono accent="plain" />
        {data.metrics?.macro_f1 != null && (
          <Tile
            label="Macro F1"
            value={data.metrics.macro_f1.toFixed(3)}
            sub="5-fold CV, out-of-fold"
            mono
            accent="plain"
          />
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <div className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-muted mb-3">
            Class probabilities
          </div>
          {data.ranked.map((c, i) => (
            <div key={c.code} className="mb-4">
              <div className="flex justify-between items-baseline text-sm mb-2">
                <span className={i === 0 ? 'font-semibold' : 'text-muted'}>
                  {c.label}
                </span>
                <span className="font-mono tabular-nums text-sm">{pct(c.prob)}</span>
              </div>
              <div className="h-2 rounded-full bg-line overflow-hidden">
                <div
                  className="h-full rounded-full transition-[width] duration-700 ease-out"
                  style={{
                    width: pct(c.prob),
                    background:
                      i === 0
                        ? 'linear-gradient(90deg,var(--color-accent),var(--color-accent3))'
                        : 'var(--color-line)',
                    boxShadow: i === 0 ? '0 0 10px rgb(245 158 11 / .4)' : 'none',
                  }}
                />
              </div>
            </div>
          ))}
          {data.threshold_flip && (
            <p className="text-[12px] leading-relaxed text-warn/90 mt-4 border-l-2 border-warn/40 pl-3">
              The averaged probability of {data.positive_class} is{' '}
              <span className="font-mono">{pct(data.positive_prob)}</span>, below 50% but
              at or above the validation-chosen threshold of{' '}
              <span className="font-mono">{pct(data.threshold)}</span> — so the call is{' '}
              {p.code} even though it is not the larger bar. The threshold, not 50%, is
              what the reported accuracy uses.
            </p>
          )}
          {data.threshold_note && (
            <p className="text-[12px] leading-relaxed text-muted mt-4 border-l-2 border-accent/40 pl-3">
              {data.threshold_note}
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3 self-start">
          <figure className="m-0">
            <figcaption className="text-[10.5px] font-semibold uppercase tracking-[0.11em] text-muted mb-2">
              {multi ? `Representative slice (#${data.representative_slice})` : 'Model input'}
            </figcaption>
            <img
              src={data.input_image}
              alt="Uploaded slice after preprocessing"
              className="w-full rounded-2xl border border-line bg-black block"
            />
          </figure>
          {data.gradcam && (
            <figure className="m-0">
              <figcaption className="text-[10.5px] font-semibold uppercase tracking-[0.11em] text-muted mb-2">
                Attention map
              </figcaption>
              <img
                src={data.gradcam}
                alt="Grad-CAM attention overlay"
                className="w-full rounded-2xl border border-line bg-black block"
              />
            </figure>
          )}
          <p className="col-span-2 text-[11.5px] leading-relaxed text-muted">
            {data.representative_note}
          </p>
          <p className="col-span-2 text-[11.5px] leading-relaxed text-muted">
            Red marks the regions that most influenced the prediction. On earlier
            versions of this model the heat sat outside the brain entirely — that is how
            the scanner confound was caught.
          </p>
        </div>
      </div>

      {data.metrics?.confusion_matrix && (
        <div className="mt-6 pt-6 border-t border-line">
          <ConfusionMatrix
            matrix={data.metrics.confusion_matrix}
            classes={data.metrics.confusion_classes}
            macroF1={data.metrics.macro_f1}
          />
        </div>
      )}
    </section>
  )
}
