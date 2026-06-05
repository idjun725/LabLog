import lablogMark from '../assets/lablog-mark.png'
import styles from './LabLogLogo.module.css'

type LabLogLogoProps = {
  className?: string
}

export function LabLogLogo({ className }: LabLogLogoProps) {
  return (
    <div className={`${styles.brand} ${className ?? ''}`} aria-label="LabLog">
      <span className={styles.wordmark}>Lab<span className={styles.wordmarkBold}>Log</span></span>
      <div className={styles.iconWrap} aria-hidden>
        <img
          className={styles.icon}
          src={lablogMark}
          alt=""
          width={840}
          height={814}
        />
      </div>
    </div>
  )
}
