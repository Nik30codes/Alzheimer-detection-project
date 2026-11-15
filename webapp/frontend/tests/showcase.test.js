import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { batchCsv } from '../src/batch.js'

test('showcase contains recorded successes and mistakes without participant IDs', () => {
  const evidence = JSON.parse(readFileSync(new URL('../../backend/research_assets/benchmark.json', import.meta.url),'utf8'))
  assert.equal(evidence.recorded_examples.length,4)
  assert.equal(evidence.recorded_examples.filter(e => e.correct).length,2)
  assert.equal(evidence.mri.metrics.subjects,501)
  for (const e of evidence.recorded_examples) {
    assert.equal(e.correct,e.reference===e.prediction)
    assert.ok(e.ad_score>=0 && e.ad_score<=1)
    assert.equal(e.subject_id,undefined)
  }
})
test('export labels fictional context and preserves absent cognitive scores', () => {
  const csv = batchCsv([{id:'Sample',status:'Queued',files:[],clinical:{age:72,sex:'female',education_years:16,mmse:null,moca:null,demo:true}}])
  assert.ok(csv.includes('Fictional demonstration values'))
  assert.ok(csv.includes('"72","female","16","",""'))
})
