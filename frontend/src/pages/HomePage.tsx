import { Link } from 'react-router-dom'
import { AppShell } from '../components/AppShell'
import { LabLogLogo } from '../components/LabLogLogo'
import logoStyles from '../components/LabLogLogo.module.css'
import styles from './HomePage.module.css'

function UploadIcon() {
  return (
    <svg
      className={styles.heroIcon}
      viewBox="0 0 64 64"
      fill="none"
      stroke="currentColor"
      strokeWidth={8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polyline points="16,25 32,8 48,25" />
      <line x1="32" y1="8" x2="32" y2="43" />
      <path d="M11,43 L11,56 L53,56 L53,43" />
    </svg>
  )
}

function CameraIcon() {
  return (
    <svg
      className={`${styles.heroIcon} ${styles.heroIconCamera}`}
      viewBox="0 0 64 64"
      aria-hidden
    >
      <path
        fillRule="evenodd"
        fill="currentColor"
        d="M12,12 H52 a10,10 0 0,1 10,10 V50 a10,10 0 0,1 -10,10 H12 a10,10 0 0,1 -10,-10 V22 a10,10 0 0,1 10,-10 Z M32,26 A11,11 0 0,0 32,48 A11,11 0 0,0 32,26 Z"
      />
      <rect x="13" y="2" width="38" height="12" rx="7" fill="currentColor" />
    </svg>
  )
}

export function HomePage() {
  return (
    <AppShell variant="home">
      <div className={styles.paperBg} aria-hidden />
      <LabLogLogo className={logoStyles.hero} />

      <div className={styles.actions}>
        <Link to="/upload" className={`${styles.btn} ${styles.btnNavy} ${styles.btnTall}`}>
          <UploadIcon />
          <span>동영상 업로드</span>
        </Link>

        <Link to="/record" className={`${styles.btn} ${styles.btnLavender} ${styles.btnTall}`}>
          <CameraIcon />
          <span>동영상 촬영</span>
        </Link>

        <Link to="/archive" className={`${styles.btn} ${styles.btnNavy} ${styles.btnShort}`}>
          보관함
        </Link>

        <Link to="/drafts" className={`${styles.btn} ${styles.btnNavy} ${styles.btnShort}`}>
          <span className={styles.badge} aria-label="새 항목 3건">
            3
          </span>
          임시 보관함
        </Link>
      </div>
    </AppShell>
  )
}
