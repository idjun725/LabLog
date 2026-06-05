import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppShell } from '../components/AppShell'
import { ExperimentInfoForm } from '../components/ExperimentInfoForm'
import { PageHeader } from '../components/PageHeader'
import styles from './UploadPage.module.css'

const ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.webm'

export function UploadPage() {
  const navigate = useNavigate()
  const [fileName, setFileName] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const goLoading = useCallback(
    (file: File) => {
      navigate('/loading', { state: { phase: 'upload', file } })
    },
    [navigate],
  )

  const onFiles = useCallback(
    (files: FileList | null) => {
      const f = files?.[0]
      if (f) {
        setFileName(f.name)
        goLoading(f)
      }
    },
    [goLoading],
  )

  return (
    <AppShell variant="page">
      <PageHeader title="동영상 업로드" />

      <p className={styles.hint}>
        MP4, MOV, AVI, WebM · 최대 2GB (MVP 기준)
      </p>

      <ExperimentInfoForm />

      <label
        className={`${styles.dropzone} ${isDragging ? styles.dropzoneActive : ''}`}
        onDragEnter={() => setIsDragging(true)}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          onFiles(e.dataTransfer.files)
        }}
      >
        <input
          type="file"
          accept={ACCEPT}
          className={styles.input}
          onChange={(e) => onFiles(e.target.files)}
        />
        <span className={styles.dropTitle}>파일을 끌어다 놓거나 탭하여 선택</span>
        <span className={styles.dropSub}>업로드 후 AI 분석이 시작됩니다</span>
      </label>

      {fileName ? <p className={styles.fileName}>{fileName}</p> : null}
    </AppShell>
  )
}
