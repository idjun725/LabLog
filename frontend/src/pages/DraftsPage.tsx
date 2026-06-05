import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listDrafts, listHiddenMockIds, type Draft } from '../api/lablog'
import styles from './DraftsPage.module.css'

const MOCK_ITEMS = [
  { id: '1', title: '보고서A', status: '초안 검수', hasClip: true },
  { id: '2', title: '보고서B', status: '초안 제작 중...', hasClip: false },
  { id: '3', title: '보고서C', status: '최종본 제작 중...', hasClip: false },
  { id: '4', title: '보고서D', status: '최종본 검수', hasClip: true },
  { id: '5', title: '보고서E', status: '최종본 검수', hasClip: true },
]

const POLL_INTERVAL_MS = 2000

function statusLabel(status: Draft['status']): string {
  switch (status) {
    case 'pending':
      return '분석 진행중...'
    case 'failed':
      return '분석 실패'
    case 'complete':
    default:
      return '분석 완료'
  }
}

function Paperclip() {
  return (
    <svg
      className={styles.clip}
      viewBox="0 0 30 72"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M22 10 C22 5 8 5 8 10 L8 52 C8 62 22 62 22 52 L22 14 C22 9 12 9 12 14 L12 48"
        stroke="#c0392b"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function DraftsPage() {
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [visibleMocks, setVisibleMocks] = useState(MOCK_ITEMS)

  useEffect(() => {
    const hidden = listHiddenMockIds()
    setVisibleMocks(MOCK_ITEMS.filter((m) => !hidden.has(m.id)))
    // 임시 보관함은 "보고서 미생성" 상태만 — 보고서가 만들어진 항목은 /archive로 이동.
    setDrafts(listDrafts().filter((d) => !d.report))

    // 분석 진행중인 draft가 있으면 주기적으로 새로 읽어와 갱신한다.
    const interval = window.setInterval(() => {
      const current = listDrafts().filter((d) => !d.report)
      setDrafts(current)
      if (!current.some((d) => d.status === 'pending')) {
        window.clearInterval(interval)
      }
    }, POLL_INTERVAL_MS)

    return () => window.clearInterval(interval)
  }, [])

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <Link to="/" className={styles.back}>‹</Link>
        <h1 className={styles.title}>임시 보관함</h1>
      </header>

      <main className={styles.content}>
        <ul className={styles.list}>
          {drafts.map((d) => (
            <li key={d.id} className={styles.cardWrap}>
              <Paperclip />
              <Link to={`/analysis/${d.id}`} className={styles.card}>
                <span className={styles.cardTitle}>{d.title}</span>
                <span className={styles.cardStatus}>({statusLabel(d.status)})</span>
              </Link>
              <div className={styles.fold} />
            </li>
          ))}
          {visibleMocks.map((item) => (
            <li key={item.id} className={styles.cardWrap}>
              {item.hasClip && <Paperclip />}
              <Link to={`/report/${item.id}`} className={styles.card}>
                <span className={styles.cardTitle}>{item.title}</span>
                <span className={styles.cardStatus}>({item.status})</span>
              </Link>
              <div className={styles.fold} />
            </li>
          ))}
        </ul>

        <div className={styles.chevronRow}>
          <svg viewBox="0 0 24 24" fill="none" className={styles.chevron}>
            <path
              d="M6 9L12 15L18 9"
              stroke="#888"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </main>
    </div>
  )
}
