import { Fragment } from 'react'

/**
 * 2x2 confusion matrix for the AD-vs-CN CV headline. `matrix` is
 * [[TN(CN->CN), FP(CN->AD)], [FN(AD->CN), TP(AD->AD)]], `classes` is ["CN","AD"] --
 * both read live from /api/config (webapp/backend/tasks.py), which in turn reads
 * reports/mobilenetv2_ADvsCN_cv_confusion.json. A plain CSS grid is enough for 2x2;
 * no chart library needed.
 */
export default function ConfusionMatrix({ matrix, classes, macroF1 }) {
  if (!matrix || !classes) return null
  const total = matrix.flat().reduce((a, b) => a + b, 0)
  const max = Math.max(...matrix.flat())

  return (
    <div>
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-[10.5px] font-semibold uppercase tracking-[0.13em] text-muted">
          Confusion matrix
        </div>
        {macroF1 != null && (
          <div className="font-mono text-[12px] text-muted">
            macro F1 <span className="font-semibold text-ink">{macroF1.toFixed(3)}</span>
          </div>
        )}
      </div>
      <div className="grid grid-cols-[auto_repeat(2,1fr)] gap-1.5 max-w-[280px]">
        <div />
        {classes.map((c) => (
          <div key={c} className="text-center text-[11px] font-semibold text-muted pb-1">
            pred {c}
          </div>
        ))}
        {matrix.map((row, r) => (
          <Fragment key={r}>
            <div className="flex items-center text-[11px] font-semibold text-muted pr-1">
              true {classes[r]}
            </div>
            {row.map((v, c) => {
              const correct = r === c
              const intensity = max ? v / max : 0
              return (
                <div
                  key={c}
                  className="rounded-xl grid place-items-center py-3 font-mono text-[15px] font-bold"
                  style={{
                    background: correct
                      ? `rgb(22 163 74 / ${0.08 + intensity * 0.28})`
                      : `rgb(220 38 38 / ${0.06 + intensity * 0.24})`,
                    color: correct ? 'var(--color-ok)' : 'var(--color-danger)',
                    border: `1px solid ${correct ? 'rgb(22 163 74 / .25)' : 'rgb(220 38 38 / .2)'}`,
                  }}
                >
                  {v}
                </div>
              )
            })}
          </Fragment>
        ))}
      </div>
      <p className="mt-2.5 text-[11px] leading-relaxed text-muted max-w-[280px]">
        {total} subjects, out-of-fold across all 5 cross-validation folds — the same
        predictions behind the accuracy/AUC tiles above.
      </p>
    </div>
  )
}
