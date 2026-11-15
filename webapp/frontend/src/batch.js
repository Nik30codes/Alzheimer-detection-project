const IMAGE = /\.(png|jpe?g|dcm)$/i

export function groupPatientFiles(fileList, maxFiles = 64) {
  const groups = new Map()
  let ignored = 0
  for (const file of Array.from(fileList)) {
    const parts = (file.webkitRelativePath || '').split('/')
    if (!IMAGE.test(file.name)) { ignored++; continue }
    if (parts.length < 3 || !parts[1]) {
      throw new Error('Choose a parent folder containing one subfolder per patient. Images directly inside the selected folder cannot be assigned to a patient.')
    }
    const id = parts[1]
    if (!groups.has(id)) groups.set(id, [])
    groups.get(id).push(file)
  }
  if (!groups.size) throw new Error('No PNG, JPEG or DICOM images found in patient subfolders.')
  if (groups.size > 100) throw new Error('Select at most 100 patient folders per batch.')
  const patients = [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([id, files]) => {
    files.sort((a, b) => a.webkitRelativePath.localeCompare(b.webkitRelativePath, undefined, { numeric: true }))
    const error = files.length > maxFiles ? `Too many images (maximum ${maxFiles} per patient).`
      : files.some(f => f.size > 8 * 1024 * 1024) ? 'An image exceeds 8 MB.'
      : files.reduce((sum, f) => sum + f.size, 0) > 48 * 1024 * 1024 ? 'Patient images exceed 48 MB in total.'
      : null
    return { id, files, status: error ? 'Failed' : 'Queued', error }
  })
  return { patients, ignored }
}

export async function readPrediction(response, onProgress = () => {}) {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}))
    throw new Error(detail.detail || `Request failed (${response.status})`)
  }
  if (!response.body) throw new Error('The server returned an empty response.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = '', result = null
  function consume(line) {
    if (!line.trim()) return
    const event = JSON.parse(line)
    if (event.error) throw new Error(event.error)
    if (event.pct != null) onProgress(event.pct, event)
    if (event.result) result = event.result
  }
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()
      lines.forEach(consume)
    }
    buffer += decoder.decode()
    consume(buffer)
    if (!result) throw new Error('The response ended before a result arrived. Retry this patient.')
    return result
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

export function patientWarnings(patient) {
  if (patient.error) return patient.error
  if (!patient.result) return ''
  const r = patient.result
  const notes = ['Folder grouping supplied by uploader; patient identity not verified.']
  if (r.n_slices_used !== 32) notes.push(`${r.n_slices_used} usable slices; evaluation used 32 per subject.`)
  if (r.n_slices_skipped) notes.push(`${r.n_slices_skipped} images rejected.`)
  if (r.research && !r.research.available) notes.push(r.research.reason)
  if (r.research?.disagreement) notes.push('Experimental models disagree; inspect the comparison.')
  notes.push('Research result, not a diagnosis. Probabilities are not clinical risk estimates.')
  return notes.join(' ')
}

export function batchCsv(patients) {
  // Quoting alone does not prevent spreadsheet formula execution.
  const escape = value => {
    let text = String(value ?? '')
    if (/^\s*[=+\-@]/.test(text)) text = "'" + text
    return `"${text.replaceAll('"', '""')}"`
  }
  const rows = [['Patient folder', 'Status', 'Images selected', 'Images used', 'Images rejected',
    'Model prediction', 'AD probability (%)', 'CN probability (%)', 'Elapsed seconds', 'Explanation method', 'Notes', 'Attention AD (%)', 'GAN-trained AD (%)', 'Research fold', 'Research comparison note', 'Supplied age', 'Supplied sex', 'Supplied education years', 'Supplied MMSE', 'Supplied MoCA', 'Patient details source']]
  for (const p of patients) {
    const r = p.result
    const probability = code => r ? ((r.ranked.find(c => c.code === code)?.prob ?? 0) * 100).toFixed(1) : ''
    rows.push([p.id, p.status, p.files.length, r?.n_slices_used, r?.n_slices_skipped,
      r?.prediction.label, probability('AD'), probability('CN'), r?.elapsed, r ? 'Grad-CAM' : '', patientWarnings(p),
      r?.research?.available ? (r.research.arms.attention.ad_probability*100).toFixed(1) : '',
      r?.research?.available ? (r.research.arms.attention_gan.ad_probability*100).toFixed(1) : '',
      r?.research?.fold, r?.research?.note || r?.research?.reason,
      p.clinical?.age, p.clinical?.sex, p.clinical?.education_years, p.clinical?.mmse, p.clinical?.moca, p.clinical ? (p.clinical.demo ? 'Fictional demonstration values' : 'Supplied by uploader; unverified') : 'Not supplied'])
  }
  return '\uFEFF' + rows.map(row => row.map(escape).join(',')).join('\r\n')
}
