import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listDrafts, listHiddenMockIds, type Draft } from '../api/lablog'
import styles from './ArchivePage.module.css'

const MOCK_ITEMS = [
  { id: 'archive-1', title: '보고서1' },
  { id: 'archive-2', title: '보고서2' },
  { id: 'archive-3', title: '보고서3' },
  { id: 'archive-4', title: '보고서4' },
  { id: 'archive-5', title: '보고서5' },
  { id: 'archive-6', title: '보고서6' },
]

export function ArchivePage() {
  const [mockItems, setMockItems] = useState(MOCK_ITEMS)
  const [archivedDrafts, setArchivedDrafts] = useState<Draft[]>([])

  useEffect(() => {
    const hidden = listHiddenMockIds()
    setMockItems(MOCK_ITEMS.filter((m) => !hidden.has(m.id)))
    // 보관함은 "보고서 생성 완료" 항목만 표시 (보고서 없는 draft는 /drafts에서 노출)
    setArchivedDrafts(listDrafts().filter((d) => !!d.report))
  }, [])

  // 실제 draft(최신순)를 위에, mock 항목을 아래에 stacked.
  const items = [
    ...archivedDrafts.map((d) => ({ id: d.id, title: d.title })),
    ...mockItems,
  ]
  const pct = items.length > 0 ? 100 / items.length : 0

  return (
    <div className={styles.shell}>
      <div className={styles.paperBg} aria-hidden />
      <header className={styles.header}>
        <Link to="/" className={styles.back}>‹</Link>
        <h1 className={styles.title}>보관함</h1>
      </header>

      <div className={styles.folderTab} />

      {items.length > 0 ? (
        <ul className={styles.list}>
          {items.map((item, i) => (
            <li
              key={item.id}
              style={{
                top: `${i * pct}%`,
                zIndex: i + 1,
                height: `${pct}%`,
              }}
            >
              <Link to={`/report/${item.id}`} className={styles.docCard}>
                <span className={styles.docTitle}>{item.title}</span>
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <div className={styles.empty}>아직 생성된 보고서가 없습니다.</div>
      )}

      <div className={styles.folderFooter}>
        <svg viewBox="0 0 24 24" fill="none" className={styles.chevron}>
          <path d="M6 9L12 15L18 9" stroke="var(--lab-white)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    </div>
  )
}
