import type { ReactNode } from 'react'
import styles from './AppShell.module.css'

type AppShellProps = {
  children: ReactNode
  variant?: 'home' | 'page'
}

export function AppShell({ children, variant = 'page' }: AppShellProps) {
  return (
    <div
      className={`${styles.shell} ${variant === 'home' ? styles.shellHome : styles.shellPage}`}
    >
      <div className={styles.inner}>{children}</div>
    </div>
  )
}
