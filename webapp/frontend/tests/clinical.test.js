import test from 'node:test'
import assert from 'node:assert/strict'
import { parseClinicalCsv } from '../src/clinical.js'

test('matches explicit identifiers, preserves missingness and quoted identifiers', () => {
  const rows = parseClinicalCsv('\uFEFFpatient_id,age,sex,education_years,mmse,moca\r\n"A,1",72,female,16,27,\r\nB,,unknown,,,22')
  assert.equal(rows.get('A,1').age, 72)
  assert.equal(rows.get('A,1').moca, null)
  assert.equal(rows.get('B').age, null)
  assert.equal(rows.get('B').moca, 22)
})
test('rejects ambiguous identifiers, leaked labels and invalid clinical values', () => {
  for (const input of ['patient_id\nA\nA', 'patient_id,diagnosis\nA,AD', 'patient_id,mmse\nA,-4', 'patient_id,moca\nA,31', 'patient_id,age\nA,=10', 'patient_id\n"A']) {
    assert.throws(() => parseClinicalCsv(input))
  }
})
