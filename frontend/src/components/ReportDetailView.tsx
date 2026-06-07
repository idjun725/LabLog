import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import type { GeneratedReport } from '../api/lablog'
import styles from './ReportDetailView.module.css'

type Props = {
  report: GeneratedReport
  cardTitle: string
  backTo: string
  onDelete: () => void
  // 옵션 — 실제 draft에만 전달. 둘 다 없으면 cardTitle은 표시만 되고 분석편집 버튼도 숨김.
  onTitleChange?: (newTitle: string) => void
  analysisEditHref?: string
}

// 단일 세로 스크롤 보고서 뷰. mock 보고서와 real draft 보고서가 같은 뷰를 공유한다.
export function ReportDetailView({
  report,
  cardTitle,
  backTo,
  onDelete,
  onTitleChange,
  analysisEditHref,
}: Props) {
  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <Link to={backTo} className={styles.back}>‹</Link>
        <div className={styles.headerCenter}>
          <CardTitle value={cardTitle} onChange={onTitleChange} />
        </div>
        <div className={styles.headerActions}>
          {analysisEditHref && (
            <Link
              to={analysisEditHref}
              className={styles.editAnalysisBtn}
              title="분석 결과 편집 / 보고서 재생성"
            >
              분석편집
            </Link>
          )}
          <button
            type="button"
            className={styles.trashBtn}
            onClick={onDelete}
            aria-label="삭제"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
              <path d="M10 11v6M14 11v6" />
              <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
            </svg>
          </button>
          <button className={styles.downloadBtn} aria-label="다운로드">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </button>
        </div>
      </header>

      <main className={styles.content}>
        <div className={styles.titleCard}>
          <span className={styles.titleBadge}>실험 제목</span>
          <div className={styles.titleValue}>{report.title}</div>
        </div>

        <Row num={1} label="실험 날짜">{report.date || '데이터 부족'}</Row>

        <Row num={2} label="선행 연구">
          <pre className={styles.pre}>{report.preliminary_research}</pre>
        </Row>

        <Row num={3} label="실험 목적">
          <pre className={styles.pre}>{report.objective}</pre>
        </Row>

        <Row num={4} label="가설">
          <pre className={styles.pre}>{report.hypothesis}</pre>
        </Row>

        <Row num={5} label="준비물">
          {report.materials.length > 0 ? report.materials.join(', ') : '데이터 부족'}
        </Row>

        <Row num={6} label="실험 방법">
          <pre className={styles.pre}>{report.method}</pre>
        </Row>

        <Row num={7} label="실험 과정">
          {report.procedure.length > 0 ? (
            report.procedure.map((p, i) => (
              <div key={i} className={styles.bullet}>{i + 1}) {p}</div>
            ))
          ) : (
            '데이터 부족'
          )}
        </Row>

        <Row num={8} label="실험 결과">
          <pre className={styles.pre}>{report.results}</pre>
        </Row>

        <Row num={9} label="결론">
          <pre className={styles.pre}>{report.conclusion}</pre>
        </Row>
      </main>
    </div>
  )
}

// 헤더 가운데에 표시되는 카드 제목. onChange가 있으면 inline 편집 가능.
function CardTitle({
  value,
  onChange,
}: {
  value: string
  onChange?: (next: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)

  const commit = useCallback(() => {
    const trimmed = draft.trim()
    setEditing(false)
    if (trimmed && trimmed !== value) onChange?.(trimmed)
    else setDraft(value)
  }, [draft, value, onChange])

  if (!onChange) {
    return <span className={styles.cardTitleStatic}>{value}</span>
  }
  if (editing) {
    return (
      <input
        className={styles.cardTitleInput}
        value={draft}
        autoFocus
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            commit()
          } else if (e.key === 'Escape') {
            setEditing(false)
            setDraft(value)
          }
        }}
      />
    )
  }
  return (
    <button
      type="button"
      className={styles.cardTitleEditable}
      onClick={() => {
        setDraft(value)
        setEditing(true)
      }}
      title="클릭하여 파일명 수정"
    >
      {value} <span className={styles.cardTitlePencil}>✎</span>
    </button>
  )
}

function Row({
  num,
  label,
  children,
}: {
  num: number
  label: string
  children: React.ReactNode
}) {
  return (
    <div className={styles.row}>
      <div className={styles.labelCol}>
        <span className={styles.badge}>{num}</span>
        <span className={styles.labelText}>{label}</span>
      </div>
      <div className={styles.valueCol}>{children}</div>
    </div>
  )
}
