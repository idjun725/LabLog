import { Link } from 'react-router-dom'
import styles from './PageHeader.module.css'

type PageHeaderProps = {
  title: string
  backTo?: string
}

export function PageHeader({ title, backTo = '/' }: PageHeaderProps) {
  return (
    <header className={styles.header}>
      <Link to={backTo} className={styles.back}>
        ←
      </Link>
      <h1 className={styles.title}>{title}</h1>
    </header>
  )
}
