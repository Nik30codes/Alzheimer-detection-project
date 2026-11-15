// Optional uploader-supplied context; never sent to the image classifier.
export const CLINICAL_FIELDS = ['patient_id', 'age', 'sex', 'education_years', 'mmse', 'moca']

export function parseClinicalCsv(text) {
  if (text.length > 1024 * 1024) throw new Error('Clinical CSV must be smaller than 1 MB.')
  const rows = []; let row = [], value = '', quoted = false, closed = false
  const source = text.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n')
  for (let i = 0; i <= source.length; i++) {
    const c = source[i]
    if (quoted) {
      if (c === undefined) throw new Error('Unclosed CSV quote.')
      if (c === '"' && source[i + 1] === '"') { value += '"'; i++ }
      else if (c === '"') { quoted = false; closed = true }
      else value += c
    } else if (c === ',' || c === '\n' || c === undefined) {
      row.push(value.trim()); value = ''; closed = false
      if (c !== ',') { if (row.some(v => v !== '')) rows.push(row); row = [] }
    } else if (c === '"') {
      if (value || closed) throw new Error('Invalid CSV quote.')
      quoted = true
    } else {
      if (closed) throw new Error('Unexpected text after CSV quote.')
      value += c
    }
  }
  const [headers, ...records] = rows
  if (!headers || !headers.includes('patient_id')) throw new Error('CSV requires a patient_id column matching the patient folder.')
  if (new Set(headers).size !== headers.length || headers.some(h => !CLINICAL_FIELDS.includes(h))) throw new Error(`Use unique columns from: ${CLINICAL_FIELDS.join(', ')}.`)
  if (!records.length || records.length > 100) throw new Error('Provide 1 to 100 clinical rows.')
  const output = new Map()
  const ranges = { age: [18, 120], education_years: [0, 40], mmse: [0, 30], moca: [0, 30] }
  records.forEach((values, index) => {
    if (values.length !== headers.length) throw new Error(`Row ${index + 2}: column count differs from header.`)
    const record = Object.fromEntries(headers.map((h, i) => [h, values[i]]))
    if (!record.patient_id || output.has(record.patient_id)) throw new Error(`Row ${index + 2}: missing or duplicate patient_id.`)
    for (const [field, [min, max]] of Object.entries(ranges)) {
      const v = record[field]
      if (v === undefined || v === '') { record[field] = null; continue }
      if (!/^\d+(\.\d+)?$/.test(v) || Number(v) < min || Number(v) > max) throw new Error(`Row ${index + 2}: ${field} must be ${min}–${max}, or blank.`)
      record[field] = Number(v)
    }
    if (record.sex && !['female','male','other','unknown'].includes(record.sex.toLowerCase())) throw new Error(`Row ${index + 2}: sex must be female, male, other, unknown, or blank.`)
    output.set(record.patient_id, record)
  })
  return output
}
