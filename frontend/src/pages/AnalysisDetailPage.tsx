import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ExperimentInfoForm } from '../components/ExperimentInfoForm'
import {
  classifyRecords,
  deleteDraft,
  generateReport,
  getDraft,
  markDraftSeen,
  updateDraftInfo,
  updateDraftRecord,
  updateDraftTitle,
  type AnalysisRecord,
  type Draft,
  type ExperimentInfo,
  type Phase,
} from '../api/lablog'
import styles from './AnalysisDetailPage.module.css'

const POLL_INTERVAL_MS = 2000

export function AnalysisDetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const [draft, setDraft] = useState<Draft | null>(() => getDraft(id))

  useEffect(() => {
    if (id) markDraftSeen(id)
  }, [id])

  const handleDelete = useCallback(() => {
    if (!window.confirm('이 분석 결과를 삭제하시겠습니까?')) return
    deleteDraft(id)
    navigate('/drafts')
  }, [id, navigate])

  const [generating, setGenerating] = useState(false)
  const [genError, setGenError] = useState<string | null>(null)

  const handleGenerateReport = useCallback(async () => {
    setGenerating(true)
    setGenError(null)
    try {
      await generateReport(id)
      navigate(`/report/${id}`)
    } catch (e) {
      setGenError((e as Error).message)
    } finally {
      setGenerating(false)
    }
  }, [id, navigate])

  const [classifying, setClassifying] = useState(false)
  const [classifyError, setClassifyError] = useState<string | null>(null)

  const handleClassify = useCallback(async () => {
    setClassifying(true)
    setClassifyError(null)
    try {
      await classifyRecords(id)
      setDraft(getDraft(id))
    } catch (e) {
      setClassifyError((e as Error).message)
    } finally {
      setClassifying(false)
    }
  }, [id])

  const handleTitleChange = useCallback(
    (newTitle: string) => {
      updateDraftTitle(id, newTitle)
      setDraft(getDraft(id))
    },
    [id],
  )

  const handleInfoChange = useCallback(
    (next: ExperimentInfo) => {
      updateDraftInfo(id, next)
      setDraft(getDraft(id))
    },
    [id],
  )

  const handleRecordChange = useCallback(
    (index: number, patch: Partial<Pick<AnalysisRecord, 'ocr' | 'speech'>>) => {
      updateDraftRecord(id, index, patch)
      setDraft(getDraft(id))
    },
    [id],
  )

  const DeleteBtn = () => (
    <button
      type="button"
      className={styles.deleteBtn}
      onClick={handleDelete}
      aria-label="삭제"
    >
      삭제
    </button>
  )

  // pending(업로드 흐름) 또는 audio_pending(촬영 흐름) 동안 주기적 갱신
  useEffect(() => {
    if (!draft) return
    const needsPolling = draft.status === 'pending' || draft.audio_pending === true
    if (!needsPolling) return
    const interval = window.setInterval(() => {
      const fresh = getDraft(id)
      setDraft((prev) => {
        if (prev === fresh) return prev
        if (prev && fresh && JSON.stringify(prev) === JSON.stringify(fresh)) {
          return prev
        }
        return fresh
      })
      const stillNeeded =
        fresh && (fresh.status === 'pending' || fresh.audio_pending === true)
      if (!stillNeeded) {
        window.clearInterval(interval)
      }
    }, POLL_INTERVAL_MS)
    return () => window.clearInterval(interval)
  }, [id, draft?.status, draft?.audio_pending])

  if (!draft) {
    return (
      <div className={styles.shell}>
        <header className={styles.header}>
          <Link to="/drafts" className={styles.back}>‹</Link>
          <h1 className={styles.title}>분석 결과</h1>
          <span className={styles.headerRight} />
        </header>
        <div className={styles.empty}>
          임시 보관함에서 분석 결과를 찾을 수 없습니다.
        </div>
      </div>
    )
  }

  if (draft.status === 'pending') {
    return (
      <div className={styles.shell}>
        <header className={styles.header}>
          <Link to="/drafts" className={styles.back}>‹</Link>
          <EditableTitle value={draft.title} onSave={handleTitleChange} />
          <DeleteBtn />
        </header>
        <div className={styles.empty}>
          분석 진행중입니다. 잠시만 기다려주십시오.
          <br />
          (백엔드에서 YOLO·OCR 추론 처리 중)
        </div>
      </div>
    )
  }

  if (draft.status === 'failed') {
    return (
      <div className={styles.shell}>
        <header className={styles.header}>
          <Link to="/drafts" className={styles.back}>‹</Link>
          <EditableTitle value={draft.title} onSave={handleTitleChange} />
          <DeleteBtn />
        </header>
        <div className={styles.empty}>
          분석 실패{draft.error ? `: ${draft.error}` : ''}
        </div>
      </div>
    )
  }

  // status === 'complete' — data 보장됨
  const { filename, frame_count, records } = draft.data!
  const phases = draft.phases
  const hasReport = !!draft.report
  // report 있으면 archive에서 왔을 가능성 — back 링크를 그쪽으로
  const backTo = hasReport ? '/archive' : '/drafts'

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <Link to={backTo} className={styles.back}>‹</Link>
        <EditableTitle value={draft.title} onSave={handleTitleChange} />
        <DeleteBtn />
      </header>

      <main className={styles.content}>
        {draft.audio_pending ? (
          <div className={styles.audioPendingBanner}>
            🎤 음성 분석 진행중... (잠시 후 자동 갱신됩니다)
          </div>
        ) : null}

        <section className={styles.summary}>
          <div className={styles.summaryRow}>
            <span className={styles.summaryLabel}>파일</span>
            <span className={styles.summaryValue}>{filename ?? '(unknown)'}</span>
          </div>
          <div className={styles.summaryRow}>
            <span className={styles.summaryLabel}>분석 프레임</span>
            <span className={styles.summaryValue}>{frame_count}개</span>
          </div>
        </section>

        <ExperimentInfoForm
          value={draft.info ?? {}}
          onChange={handleInfoChange}
          title="실험 기본 정보 (수정 가능)"
          subtitle="입력한 내용은 보고서 생성 시 함께 전달됩니다. 언제든 수정/추가할 수 있습니다."
        />

        <div className={styles.reportRow}>
          <button
            type="button"
            className={styles.classifyBtn}
            onClick={handleClassify}
            disabled={classifying}
          >
            {classifying
              ? '단계 분류 중...'
              : phases
                ? '단계 재분류 (GRU)'
                : 'GRU로 단계 분류'}
          </button>
          {classifyError ? <p className={styles.genError}>{classifyError}</p> : null}

          <button
            type="button"
            className={styles.generateBtn}
            onClick={handleGenerateReport}
            disabled={generating}
          >
            {generating
              ? '보고서 생성 중...'
              : hasReport
                ? '보고서 재생성'
                : '보고서 생성'}
          </button>
          {genError ? <p className={styles.genError}>{genError}</p> : null}
        </div>

        <p className={styles.note}>
          아래 OCR·음성 텍스트는 수정 가능합니다. 필요한 부분만 고치고 "보고서{hasReport ? ' 재' : ' '}생성"을 다시 누르면 갱신됩니다.
        </p>

        <ol className={styles.records}>
          {records.map((r, i) => (
            <RecordCard
              key={i}
              index={i}
              record={r}
              phase={phases?.[i]}
              onChange={handleRecordChange}
            />
          ))}
        </ol>
      </main>
    </div>
  )
}

// 클릭하면 input으로 변신하는 inline-editable title.
// Enter로 저장, Escape로 취소. blur도 저장으로 처리.
function EditableTitle({
  value,
  onSave,
}: {
  value: string
  onSave: (next: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draftValue, setDraftValue] = useState(value)

  const commit = useCallback(() => {
    const trimmed = draftValue.trim()
    setEditing(false)
    if (trimmed && trimmed !== value) onSave(trimmed)
    else setDraftValue(value) // 빈 값/무변경이면 원래 값으로 복원
  }, [draftValue, onSave, value])

  if (!editing) {
    return (
      <button
        type="button"
        className={styles.titleEditable}
        onClick={() => {
          setDraftValue(value)
          setEditing(true)
        }}
        title="클릭하여 제목 수정"
      >
        {value} <span className={styles.titlePencil}>✎</span>
      </button>
    )
  }
  return (
    <input
      className={styles.titleInput}
      value={draftValue}
      autoFocus
      onChange={(e) => setDraftValue(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        } else if (e.key === 'Escape') {
          setEditing(false)
          setDraftValue(value)
        }
      }}
    />
  )
}

function RecordCard({
  index,
  record,
  phase,
  onChange,
}: {
  index: number
  record: AnalysisRecord
  phase?: Phase
  onChange: (
    i: number,
    patch: Partial<Pick<AnalysisRecord, 'ocr' | 'speech'>>,
  ) => void
}) {
  const [editing, setEditing] = useState(false)
  const [ocrDraft, setOcrDraft] = useState(record.ocr)
  const [speechDraft, setSpeechDraft] = useState(record.speech)
  const objects = Object.entries(record.yolo).sort((a, b) => b[1] - a[1])

  const startEdit = () => {
    setOcrDraft(record.ocr)
    setSpeechDraft(record.speech)
    setEditing(true)
  }
  const cancelEdit = () => setEditing(false)
  const saveEdit = () => {
    onChange(index, { ocr: ocrDraft, speech: speechDraft })
    setEditing(false)
  }

  return (
    <li className={styles.record}>
      <div className={styles.recordHead}>
        <span className={styles.recordTime}>{record.timestamp}</span>
        <span className={styles.recordMeta}>
          {phase ? (
            <span className={styles.phaseBadge}>{phase.phase}</span>
          ) : null}
          avg {record.avg_yolo_confidence.toFixed(2)} · bright {record.brightness.toFixed(0)}
          {editing ? null : (
            <button
              type="button"
              className={styles.editBtn}
              onClick={startEdit}
              title="OCR·음성 수정"
            >
              ✎ 수정
            </button>
          )}
        </span>
      </div>
      {record.assist ? (
        <div className={styles.assist}>{record.assist}</div>
      ) : null}
      {objects.length > 0 ? (
        <div className={styles.field}>
          <span className={styles.fieldLabel}>탐지</span>
          <div className={styles.chips}>
            {objects.map(([label, conf]) => (
              <span key={label} className={styles.chip}>
                {label} <small>{conf.toFixed(2)}</small>
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {editing ? (
        <>
          <div className={styles.field}>
            <span className={styles.fieldLabel}>OCR</span>
            <textarea
              className={styles.editTextarea}
              value={ocrDraft}
              onChange={(e) => setOcrDraft(e.target.value)}
              placeholder="화면 인식 텍스트"
            />
          </div>
          <div className={styles.field}>
            <span className={styles.fieldLabel}>음성</span>
            <textarea
              className={styles.editTextarea}
              value={speechDraft}
              onChange={(e) => setSpeechDraft(e.target.value)}
              placeholder="음성 인식 텍스트"
            />
          </div>
          <div className={styles.editActions}>
            <button type="button" className={styles.editCancel} onClick={cancelEdit}>
              취소
            </button>
            <button type="button" className={styles.editSave} onClick={saveEdit}>
              저장
            </button>
          </div>
        </>
      ) : (
        <>
          {record.ocr ? (
            <div className={styles.field}>
              <span className={styles.fieldLabel}>OCR</span>
              <span className={styles.ocrValue}>
                {record.ocr} <small>({record.ocr_confidence.toFixed(2)})</small>
              </span>
            </div>
          ) : null}
          {record.speech ? (
            <div className={styles.field}>
              <span className={styles.fieldLabel}>음성</span>
              <span className={styles.ocrValue}>{record.speech}</span>
            </div>
          ) : null}
          {objects.length === 0 && !record.ocr && !record.speech ? (
            <div className={styles.emptyRecord}>탐지된 객체·텍스트·음성 없음</div>
          ) : null}
        </>
      )}
    </li>
  )
}
