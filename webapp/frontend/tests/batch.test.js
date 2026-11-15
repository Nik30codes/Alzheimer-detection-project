import test from 'node:test'
import assert from 'node:assert/strict'
import { batchCsv, groupPatientFiles, patientWarnings, readPrediction } from '../src/batch.js'

const file = (path, size = 100) => ({ name: path.split('/').pop(), webkitRelativePath: path, size })

test('groups by patient folder, including nested scan folders, never by slice filename', () => {
  const { patients, ignored } = groupPatientFiles([
    file('Batch/B/scan/slice2.png'), file('Batch/A/slice1.png'),
    file('Batch/B/scan/slice1.png'), file('Batch/A/notes.txt'),
  ])
  assert.equal(ignored, 1)
  assert.deepEqual(patients.map(p => [p.id, p.files.length]), [['A', 1], ['B', 2]])
  assert.equal(patients[1].files[0].name, 'slice1.png')
})

test('rejects ambiguous root images and empty folders', () => {
  assert.throws(() => groupPatientFiles([file('Batch/scan.png')]), /one subfolder per patient/)
  assert.throws(() => groupPatientFiles([file('Batch/A/notes.txt')]), /No PNG/)
})

test('oversized patients fail locally without blocking valid patients', () => {
  const { patients } = groupPatientFiles([file('Batch/A/scan.png', 8 * 1024 * 1024 + 1), file('Batch/B/scan.png')])
  assert.equal(patients[0].status, 'Failed')
  assert.equal(patients[1].status, 'Queued')
  assert.equal(groupPatientFiles(Array.from({ length: 65 }, (_, i) => file(`Batch/A/${i}.png`))).patients[0].status, 'Failed')
  assert.equal(groupPatientFiles(Array.from({ length: 7 }, (_, i) => file(`Batch/A/${i}.png`, 8 * 1024 * 1024))).patients[0].status, 'Failed')
})

test('decodes chunked NDJSON and accepts a final result without trailing newline', async () => {
  const bytes = new TextEncoder().encode('{"pct":50}\n{"result":{"label":"résultat"}}')
  const progress = []
  const response = new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 3) controller.enqueue(bytes.slice(i, i + 3))
      controller.close()
    },
  }))
  assert.deepEqual(await readPrediction(response, p => progress.push(p)), { label: 'résultat' })
  assert.deepEqual(progress, [50])
})

test('does not report incomplete, failed, or malformed streams as complete', async () => {
  await assert.rejects(readPrediction(new Response('{"pct":50}\n')), /before a result/)
  await assert.rejects(readPrediction(new Response('{"error":"Invalid scan"}\n')), /Invalid scan/)
  await assert.rejects(readPrediction(new Response('broken')), SyntaxError)
  await assert.rejects(readPrediction(new Response('{"detail":"Too large"}', { status: 413 })), /Too large/)
})

test('CSV preserves text and neutralizes spreadsheet formulas', () => {
  const csv = batchCsv([{ id: '=HYPERLINK("bad")', files: [file('x')], status: 'Failed', error: 'bad,\nimage' }])
  assert.ok(csv.includes(`"'=HYPERLINK(""bad"")"`))
  assert.ok(csv.includes('"bad,\nimage"'))
  assert.ok(csv.includes('"AD probability (%)"'))
})

test('warnings explain incomplete slice sets and do not infer patient history', () => {
  const notes = patientWarnings({ result: { n_slices_used: 1, n_slices_skipped: 2 } })
  assert.match(notes, /identity not verified/)
  assert.match(notes, /evaluation used 32/)
  assert.match(notes, /2 images rejected/)
  assert.match(notes, /not a diagnosis/)
})
