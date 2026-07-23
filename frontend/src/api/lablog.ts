const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
export const WS_BASE = API_BASE.replace(/^http/, 'ws')

export type Detection = {
  label: string
  confidence: number
  box: [number, number, number, number]  // x1, y1, x2, y2 (frame coords)
}

export type AnalysisRecord = {
  timestamp: string
  yolo: Record<string, number>
  detections: Detection[]
  ocr: string
  ocr_confidence: number
  speech: string
  avg_yolo_confidence: number
  brightness: number
  frame_width: number
  frame_height: number
  assist?: string[]  // 실시간 영상 보조 안내 메시지 (저신뢰/저밝기 조건 충족 시)
}

export type VideoAnalysisResponse = {
  filename: string | null
  frame_count: number
  records: AnalysisRecord[]
}

// /ws/transcribe 메시지: Groq Whisper로 전사된 청크 결과.
// skipped=true는 RMS 임계값 미만이라 Whisper 호출 자체를 스킵한 침묵 청크.
export type TranscribeChunkMessage = {
  chunk_index: number
  elapsed_sec: number
  text: string
  skipped: boolean
}

// ─── TTS (Edge TTS) ─────────────────────────────────────────────────
// 텍스트 → mp3 blob. 실패 시 null (서버 503/500 모두 동일 처리 — 음성 안내는 graceful).
export async function synthesizeTTS(text: string): Promise<Blob | null> {
  const trimmed = text.trim()
  if (!trimmed) return null
  try {
    const res = await fetch(`${API_BASE}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: trimmed }),
    })
    if (!res.ok) return null
    return await res.blob()
  } catch {
    return null
  }
}

// ─── 임시 보관함 (localStorage 기반) ─────────────────────────────────

export type DraftStatus = 'pending' | 'complete' | 'failed'

export type GeneratedReport = {
  title: string
  date: string
  preliminary_research: string
  objective: string
  hypothesis: string
  materials: string[]
  method: string
  procedure: string[]
  results: string
  conclusion: string
}

export type Phase = {
  timestamp: string
  phase: string  // "준비" | "측정" | "반응" | "관찰" | "정리"
}

// 사용자가 업로드/촬영 전 입력하는 실험 기본 정보. 모든 필드는 선택사항.
// 한 항목도 없으면 보고서 생성 API에 info 자체를 보내지 않는다 (백엔드가 None 처리).
export type ExperimentInfo = {
  title?: string
  subject?: string
  date?: string        // YYYY-MM-DD (HTML <input type=date> 그대로)
  hypothesis?: string
  other?: string
  // best.pt 25개 클래스 밖 사물을 잡기 위한 YOLO-World 프롬프트 목록 (UploadPage 전용).
  // 직접 추가한 항목은 한글일 수 있음 — /api/analyze/video에서 백엔드가
  // 번역(class_translator.py) 후 사용.
  customClasses?: string[]
  // best.pt 25개 클래스 중 이 목록에 있는 것만 탐지하도록 제한 (영문 클래스명,
  // vectorizer.YOLO_VOCAB과 동일한 값). 비어있으면 25개 전부 탐지(기존 동작 그대로).
  allowedClasses?: string[]
}

// ExperimentInfo의 배열 필드들 — 문자열 필드와 달리 .trim()이 없어 별도 처리 필요.
const ARRAY_FIELDS = ['customClasses', 'allowedClasses'] as const

// info의 문자열 필드·배열 필드 중 하나라도 값이 있는지 확인.
export function experimentInfoHasAny(info: ExperimentInfo): boolean {
  const hasString = Object.entries(info).some(
    ([k, v]) =>
      !(ARRAY_FIELDS as readonly string[]).includes(k) &&
      typeof v === 'string' &&
      v.trim(),
  )
  const hasArray = ARRAY_FIELDS.some((k) => {
    const v = info[k]
    return Array.isArray(v) && v.length > 0
  })
  return hasString || hasArray
}

export type Draft = {
  id: string
  createdAt: number  // unix ms
  source: 'upload' | 'record'
  title: string
  status: DraftStatus
  data: VideoAnalysisResponse | null  // pending 동안엔 null
  error?: string
  report?: GeneratedReport  // LLM으로 생성된 구조화 보고서 (있을 때만)
  phases?: Phase[]          // GRU 단계 분류 결과 (있을 때만)
  audio_pending?: boolean   // RecordPage 흐름에서 백그라운드 STT 진행 중
  info?: ExperimentInfo     // 업로드/촬영 전 사용자가 입력한 기본 정보 (선택)
}

// 폼이 즉시 작성 중인 값을 localStorage에 보관. 페이지 이동 후 돌아와도 유지.
// Draft가 생성되면 그 시점의 값이 draft.info로 복사된다 (영구 저장).
const EXPERIMENT_INFO_KEY = 'lablog:experimentInfo'

export function readPendingExperimentInfo(): ExperimentInfo {
  try {
    const raw = localStorage.getItem(EXPERIMENT_INFO_KEY)
    return raw ? (JSON.parse(raw) as ExperimentInfo) : {}
  } catch {
    return {}
  }
}

export function writePendingExperimentInfo(info: ExperimentInfo): void {
  // 빈 문자열·undefined만 있으면 키 제거 — 깨끗한 상태 유지
  const cleaned: ExperimentInfo = {}
  for (const [k, v] of Object.entries(info)) {
    const key = k as keyof ExperimentInfo
    if ((ARRAY_FIELDS as readonly string[]).includes(k)) {
      if (Array.isArray(v) && v.length > 0) cleaned[key] = v as never
    } else if (typeof v === 'string' && v.trim()) {
      cleaned[key] = v.trim() as never
    }
  }
  if (Object.keys(cleaned).length === 0) {
    localStorage.removeItem(EXPERIMENT_INFO_KEY)
  } else {
    localStorage.setItem(EXPERIMENT_INFO_KEY, JSON.stringify(cleaned))
  }
}

export function clearPendingExperimentInfo(): void {
  localStorage.removeItem(EXPERIMENT_INFO_KEY)
}

// ─── Draft 제목 자동 생성 (YYMMDD-N 형식) ─────────────────────────────
// 사용자가 별도 제목 지정 안 하면 오늘 날짜 + 순번 (예: 260605-1, 260605-2).
// 순번 카운터는 일별로 localStorage에 보관해 deletion에 영향받지 않게 한다.
const TITLE_SEQUENCE_KEY = 'lablog:titleSequenceByDay'

function todayYYMMDD(date: Date = new Date()): string {
  const yy = String(date.getFullYear() % 100).padStart(2, '0')
  const mm = String(date.getMonth() + 1).padStart(2, '0')
  const dd = String(date.getDate()).padStart(2, '0')
  return `${yy}${mm}${dd}`
}

export function generateDraftTitle(): string {
  const day = todayYYMMDD()
  const raw = localStorage.getItem(TITLE_SEQUENCE_KEY)
  const data: Record<string, number> = raw ? JSON.parse(raw) : {}
  const n = (data[day] ?? 0) + 1
  data[day] = n
  localStorage.setItem(TITLE_SEQUENCE_KEY, JSON.stringify(data))
  return `${day}-${n}`
}

const DRAFTS_KEY = 'lablog:drafts'

function readDrafts(): Draft[] {
  try {
    const raw = localStorage.getItem(DRAFTS_KEY)
    if (!raw) return []
    return JSON.parse(raw) as Draft[]
  } catch {
    return []
  }
}

function writeDrafts(drafts: Draft[]): void {
  localStorage.setItem(DRAFTS_KEY, JSON.stringify(drafts))
}

export function updateDraft(
  id: string,
  patch: Partial<Omit<Draft, 'id' | 'createdAt'>>,
): void {
  const drafts = readDrafts()
  const idx = drafts.findIndex((d) => d.id === id)
  if (idx === -1) return
  drafts[idx] = { ...drafts[idx], ...patch }
  writeDrafts(drafts)
}

export function addDraft(opts: {
  source: 'upload' | 'record'
  title?: string  // 미지정 시 generateDraftTitle()이 자동 생성 (YYMMDD-N)
  data: VideoAnalysisResponse
  info?: ExperimentInfo
}): string {
  const id = crypto.randomUUID()
  const title = opts.title ?? generateDraftTitle()
  const draft: Draft = {
    ...opts,
    title,
    id,
    createdAt: Date.now(),
    status: 'complete',
  }
  const drafts = readDrafts()
  drafts.unshift(draft)
  writeDrafts(drafts)
  return id
}

// ─── Draft 부분 편집 helpers ──────────────────────────────────────────
// 제목·records·info를 분리해 편집할 수 있게 한다. UI에서 즉시 갱신용.

export function updateDraftTitle(id: string, newTitle: string): void {
  const trimmed = newTitle.trim()
  if (!trimmed) return  // 빈 제목은 무시 (기존 유지)
  updateDraft(id, { title: trimmed })
}

export function updateDraftInfo(id: string, info: ExperimentInfo): void {
  // 빈 항목만 있으면 info 자체를 제거 (보고서 API에 빈 객체 안 보내도록)
  updateDraft(id, { info: experimentInfoHasAny(info) ? info : undefined })
}

export function updateDraftRecord(
  id: string,
  recordIndex: number,
  patch: Partial<Pick<AnalysisRecord, 'ocr' | 'speech'>>,
): void {
  const draft = getDraft(id)
  if (!draft || !draft.data) return
  if (recordIndex < 0 || recordIndex >= draft.data.records.length) return
  const updatedRecords = draft.data.records.map((r, i) =>
    i === recordIndex ? { ...r, ...patch } : r,
  )
  updateDraft(id, { data: { ...draft.data, records: updatedRecords } })
}

export function listDrafts(): Draft[] {
  return readDrafts().sort((a, b) => b.createdAt - a.createdAt)
}

export function getDraft(id: string): Draft | null {
  return readDrafts().find((d) => d.id === id) ?? null
}

export function deleteDraft(id: string): void {
  writeDrafts(readDrafts().filter((d) => d.id !== id))
}

// ─── Claude 보고서 생성 ───────────────────────────────────────────────
// 백엔드 /api/report/generate를 호출해 records → 구조화 보고서를 받는다.
// 성공 시 draft.report에 저장. 실패 시 호출자에게 명확한 오류를 throw.

export async function generateReport(draftId: string): Promise<GeneratedReport> {
  const draft = getDraft(draftId)
  if (!draft) throw new Error('Draft를 찾을 수 없습니다.')
  if (!draft.data) throw new Error('분석이 아직 완료되지 않았습니다.')

  // info에 한 항목이라도 있을 때만 전송 (백엔드는 None/빈 dict 모두 graceful 처리)
  const infoHasAny = draft.info && experimentInfoHasAny(draft.info)
  const res = await fetch(`${API_BASE}/api/report/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename: draft.data.filename,
      records: draft.data.records,
      phases: draft.phases ?? null,  // GRU 분류 결과가 있으면 LLM에 컨텍스트로 전달
      info: infoHasAny ? draft.info : null,
    }),
  })
  if (!res.ok) {
    let detail = ''
    try {
      const j = (await res.json()) as { detail?: string }
      detail = j.detail ?? ''
    } catch {
      /* JSON 파싱 실패 시 detail 빈 채로 */
    }
    throw new Error(
      `보고서 생성 실패 (${res.status})${detail ? ': ' + detail : ''}`,
    )
  }

  const report = (await res.json()) as GeneratedReport
  updateDraft(draftId, { report })
  return report
}

// ─── GRU 단계 분류 ────────────────────────────────────────────────────
// 백엔드 /api/classify를 호출해 records → 시점별 단계 라벨을 받는다.
// 학습되지 않은 경우 503 + 명확한 안내 메시지 반환 (가짜 결과 X).

export async function classifyRecords(draftId: string): Promise<Phase[]> {
  const draft = getDraft(draftId)
  if (!draft) throw new Error('Draft를 찾을 수 없습니다.')
  if (!draft.data) throw new Error('분석이 아직 완료되지 않았습니다.')

  const res = await fetch(`${API_BASE}/api/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records: draft.data.records }),
  })
  if (!res.ok) {
    let detail = ''
    try {
      const j = (await res.json()) as { detail?: string }
      detail = j.detail ?? ''
    } catch {
      /* JSON 파싱 실패 시 detail 빈 채로 */
    }
    throw new Error(
      `단계 분류 실패 (${res.status})${detail ? ': ' + detail : ''}`,
    )
  }

  const body = (await res.json()) as { phases: Phase[] }
  updateDraft(draftId, { phases: body.phases })
  return body.phases
}

// ─── 녹화 오디오 → STT (RecordPage 흐름) ───────────────────────────────
// 촬영 종료 시 누적된 오디오 blob을 백엔드로 보내 전사하고,
// 결과를 기존 records의 speech 필드에 매칭해서 draft를 갱신한다.

export async function transcribeRecording(
  draftId: string,
  audioBlob: Blob,
): Promise<void> {
  const draft = getDraft(draftId)
  if (!draft) throw new Error('Draft를 찾을 수 없습니다.')
  if (!draft.data) throw new Error('records가 없습니다.')

  const form = new FormData()
  form.append('audio', audioBlob, 'audio.webm')
  form.append('records', JSON.stringify(draft.data.records))

  const res = await fetch(`${API_BASE}/api/record/transcribe`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    let detail = ''
    try {
      const j = (await res.json()) as { detail?: string }
      detail = j.detail ?? ''
    } catch {
      /* ignore */
    }
    throw new Error(
      `오디오 전사 실패 (${res.status})${detail ? ': ' + detail : ''}`,
    )
  }
  const body = (await res.json()) as { records: AnalysisRecord[] }
  const fresh = getDraft(draftId)
  if (!fresh || !fresh.data) return  // 사용자가 삭제했을 수 있음
  updateDraft(draftId, {
    audio_pending: false,
    data: { ...fresh.data, records: body.records },
  })
}

// ─── draft "본 적 있음" 표시 ──────────────────────────────────────
// HomePage 임시 보관함 버튼의 빨간 배지에 미확인 draft 개수를 띄우기 위해 사용.
// markDraftSeen은 AnalysisDetailPage 진입 시 호출, getUnseenDraftCount는 HomePage 진입 시 호출.

const SEEN_DRAFTS_KEY = 'lablog:seen-drafts'

function readSeenDraftIds(): Set<string> {
  try {
    const raw = localStorage.getItem(SEEN_DRAFTS_KEY)
    return new Set<string>(raw ? JSON.parse(raw) : [])
  } catch {
    return new Set()
  }
}

export function markDraftSeen(id: string): void {
  const seen = readSeenDraftIds()
  if (seen.has(id)) return
  seen.add(id)
  localStorage.setItem(SEEN_DRAFTS_KEY, JSON.stringify([...seen]))
}

export function isDraftSeen(id: string): boolean {
  return readSeenDraftIds().has(id)
}

// 보고서 미생성 상태(즉 /drafts에 나타나는) 진짜 draft 중 아직 안 본 것 수.
// mock 항목은 카운트하지 않는다.
export function getUnseenDraftCount(): number {
  const seen = readSeenDraftIds()
  return readDrafts().filter((d) => !d.report && !seen.has(d.id)).length
}

// ─── mock 보고서 숨김 처리 ───────────────────────────────────────────
// DraftsPage / ArchivePage의 mock 항목은 코드에 하드코딩되어 있으므로,
// 사용자가 '삭제'하면 숨김 id 집합에 추가해 리스트에서 가린다.

const HIDDEN_MOCKS_KEY = 'lablog:hidden-mocks'

export function hideMockReport(id: string): void {
  const raw = localStorage.getItem(HIDDEN_MOCKS_KEY)
  const list: string[] = raw ? JSON.parse(raw) : []
  if (!list.includes(id)) {
    list.push(id)
    localStorage.setItem(HIDDEN_MOCKS_KEY, JSON.stringify(list))
  }
}

export function listHiddenMockIds(): Set<string> {
  const raw = localStorage.getItem(HIDDEN_MOCKS_KEY)
  return new Set<string>(raw ? JSON.parse(raw) : [])
}

// ─── 업로드 + 백그라운드 분석 ─────────────────────────────────────────
// 바이트 전송이 끝나면 즉시 pending draft를 만들고 onUploadComplete를 호출한다.
// XHR은 호출 컴포넌트의 라이프사이클과 무관하게 계속 응답을 기다리며,
// 응답 도착 시 localStorage의 draft를 complete/failed로 갱신한다.

export function uploadAndAnalyze(
  file: File,
  options: {
    sampleFps?: number
    info?: ExperimentInfo  // pending draft에 함께 저장돼 보고서 생성 시 사용됨
    source?: 'upload' | 'record'  // RecordPage가 녹화본을 같은 경로로 보낼 때 'record'
    onProgress?: (pct: number) => void
    onUploadComplete?: (draftId: string) => void
    onUploadError?: (error: string) => void
  } = {},
): void {
  const { sampleFps = 1, info, source = 'upload', onProgress, onUploadComplete, onUploadError } = options

  const draftId = crypto.randomUUID()
  let draftCreated = false

  const xhr = new XMLHttpRequest()
  xhr.open('POST', `${API_BASE}/api/analyze/video?sample_fps=${sampleFps}`)

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable && onProgress) {
      onProgress((e.loaded / e.total) * 100)
    }
  }

  // 바이트 전송 완료 → pending draft 생성 + 콜백 (사용자 navigate 시그널)
  xhr.upload.onload = () => {
    if (draftCreated) return
    draftCreated = true

    // YYMMDD-N 형식 자동 생성. 사용자가 분석 결과 페이지에서 언제든 수정 가능.
    const title = generateDraftTitle()

    const drafts = readDrafts()
    drafts.unshift({
      id: draftId,
      createdAt: Date.now(),
      source,
      title,
      status: 'pending',
      data: null,
      info,  // 입력된 게 없으면 undefined — draft.info 미설정 (보고서 생성 시 그래도 graceful)
    })
    writeDrafts(drafts)
    onUploadComplete?.(draftId)
  }

  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      try {
        const data = JSON.parse(xhr.responseText) as VideoAnalysisResponse
        if (draftCreated) {
          updateDraft(draftId, { status: 'complete', data })
        }
      } catch {
        if (draftCreated) {
          updateDraft(draftId, { status: 'failed', error: '응답 파싱 실패' })
        }
      }
    } else {
      const msg = `서버 응답 오류 (${xhr.status})`
      if (draftCreated) {
        updateDraft(draftId, { status: 'failed', error: msg })
      } else {
        onUploadError?.(msg)
      }
    }
  }

  xhr.onerror = () => {
    if (draftCreated) {
      updateDraft(draftId, { status: 'failed', error: '네트워크 오류' })
    } else {
      onUploadError?.('네트워크 오류')
    }
  }

  xhr.onabort = () => {
    if (draftCreated) {
      updateDraft(draftId, { status: 'failed', error: '업로드가 중단되었습니다.' })
    } else {
      onUploadError?.('업로드가 중단되었습니다.')
    }
  }

  const form = new FormData()
  form.append('file', file)
  if (info?.customClasses?.length) {
    // 백엔드는 쉼표구분 문자열로 받음 (analyze_video의 custom_classes Form 필드)
    form.append('custom_classes', info.customClasses.join(','))
  }
  if (info?.allowedClasses?.length) {
    form.append('allowed_classes', info.allowedClasses.join(','))
  }
  xhr.send(form)
}
