/**
 * Fixed decorative background: soft warm colour washes behind the page.
 *
 * The hero's brain visual is now the Three.js slice-stack (SliceStack.jsx), so this
 * stays purely atmospheric — restrained, low-opacity, and aria-hidden.
 */
export default function BrainBackground() {
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(900px 620px at 85% -10%, rgba(245,158,11,.14), transparent 60%),' +
            'radial-gradient(760px 520px at -5% 20%, rgba(251,146,60,.10), transparent 60%),' +
            'radial-gradient(700px 600px at 50% 110%, rgba(217,119,6,.08), transparent 60%)',
        }}
      />
    </div>
  )
}
