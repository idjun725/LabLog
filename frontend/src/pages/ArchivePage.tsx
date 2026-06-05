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

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <Link to="/" className={styles.back}>‹</Link>
        <h1 className={styles.title}>보관함</h1>
      </header>

      <ul className={styles.list}>
        {archivedDrafts.map((d) => (
          <li key={d.id}>
            <Link to={`/report/${d.id}`} className={styles.docCard}>
              <span className={styles.docTitle}>{d.title}</span>
              <span className={styles.arrow}>›</span>
            </Link>
          </li>
        ))}
        {mockItems.map((item) => (
          <li key={item.id}>
            <Link to={`/report/${item.id}`} className={styles.docCard}>
              <span className={styles.docTitle}>{item.title}</span>
              <span className={styles.arrow}>›</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
