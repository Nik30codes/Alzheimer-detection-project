import { Suspense, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { DoubleSide, PlaneGeometry } from 'three'

const EDGE_GEOMETRY = new PlaneGeometry(2.15, 1.7)

/**
 * The hero's 3D piece: a stack of 32 translucent planes, one per axial slice this
 * project's real pipeline extracts (48-92mm below the vertex, per docs/PROJECT_EXPLANATION.md
 * §9) -- not generic decoration. Gentle auto-rotation, a small tilt toward the pointer,
 * amber glow consistent with the rest of the page.
 */
const N_SLICES = 32

function Stack({ pointer }) {
  const group = useRef(null)
  const planes = useMemo(
    () =>
      Array.from({ length: N_SLICES }, (_, i) => {
        const t = i / (N_SLICES - 1)
        return { i, y: (t - 0.5) * 3.1, glow: 0.35 + 0.5 * Math.sin(t * Math.PI) }
      }),
    [],
  )

  useFrame((state, dt) => {
    if (!group.current) return
    group.current.rotation.y += dt * 0.18
    const targetX = -pointer.current.y * 0.28
    const targetZ = pointer.current.x * 0.22
    group.current.rotation.x += (targetX - group.current.rotation.x) * 0.05
    group.current.rotation.z += (targetZ - group.current.rotation.z) * 0.05
  })

  return (
    <group ref={group} rotation={[0.32, 0.5, 0]}>
      {planes.map(({ i, y, glow }) => (
        <mesh key={i} position={[0, y, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[2.15, 1.7]} />
          <meshBasicMaterial
            color={i % 8 === 0 ? '#F59E0B' : '#D97706'}
            transparent
            opacity={0.05 + glow * 0.13}
            side={DoubleSide}
          />
        </mesh>
      ))}
      {/* Band edges, so the stack reads as a measured volume rather than a blur. */}
      {[0, N_SLICES - 1].map((i) => {
        const t = i / (N_SLICES - 1)
        const y = (t - 0.5) * 3.1
        return (
          <lineSegments key={`edge-${i}`} position={[0, y, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <edgesGeometry args={[EDGE_GEOMETRY]} />
            <lineBasicMaterial color="#F59E0B" transparent opacity={0.5} />
          </lineSegments>
        )
      })}
    </group>
  )
}

export default function SliceStack() {
  const pointer = useRef({ x: 0, y: 0 })
  const [ready, setReady] = useState(false)

  return (
    <div
      className="relative h-[280px] sm:h-[340px] lg:h-[400px] w-full"
      onMouseMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        pointer.current = {
          x: ((e.clientX - r.left) / r.width) * 2 - 1,
          y: ((e.clientY - r.top) / r.height) * 2 - 1,
        }
      }}
    >
      <Canvas
        dpr={[1, 1.75]}
        camera={{ position: [0, 0.6, 5.4], fov: 40 }}
        onCreated={() => setReady(true)}
        gl={{ alpha: true, antialias: true }}
      >
        <Suspense fallback={null}>
          <Stack pointer={pointer} />
        </Suspense>
      </Canvas>
      {!ready && (
        <div className="absolute inset-0 grid place-items-center text-[12px] text-muted">
          Loading visualisation…
        </div>
      )}
      <div className="pointer-events-none absolute -bottom-1 left-1/2 -translate-x-1/2 text-center text-[11px] font-medium tracking-wide text-muted">
        32 axial slices · 48–92&nbsp;mm below the skull vertex
      </div>
    </div>
  )
}
