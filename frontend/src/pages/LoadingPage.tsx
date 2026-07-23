import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { AppShell } from '../components/AppShell'
import { PageHeader } from '../components/PageHeader'
import lablogMark from '../assets/lablog-mark.png'
import { experimentInfoHasAny, readPendingExperimentInfo, uploadAndAnalyze } from '../api/lablog'
import styles from './LoadingPage.module.css'

type LocationState = {
  phase?: 'upload' | 'analyze'
  file?: File
  source?: 'upload' | 'record'  // RecordPage가 녹화 후 이 페이지로 넘어올 때 'record'
}

const R = 90
const CIRC = 2 * Math.PI * R

export function LoadingPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const state = (location.state ?? null) as LocationState | null
  const file = state?.file
  const phase = state?.phase ?? 'analyze'
  const source = state?.source ?? 'upload'

  const [progress, setProgress] = useState(0)

  // 파일이 없는 경우(예: 'analyze' phase 진입)에는 원본의 가짜 진행률 애니메이션 유지.
  useEffect(() => {
    if (file) return
    const id = window.setInterval(() => {
      setProgress((p) => (p >= 92 ? 92 : p + Math.random() * 12))
    }, 400)
    return () => window.clearInterval(id)
  }, [file])

  // 실제 업로드는 백그라운드로 실행. 진행률 0~100%는 실제 바이트 전송량.
  // 업로드 완료 시점에 pending draft가 만들어지고 임시보관함으로 이동한다.
  // 그 뒤 서버의 추론은 백그라운드에서 진행되며, 응답이 도착하면 draft가 갱신된다.
  const uploadStartedRef = useRef(false)
  useEffect(() => {
    if (!file || uploadStartedRef.current) return
    uploadStartedRef.current = true

    // 사용자가 UploadPage에서 입력한 기본 정보(있으면)를 함께 첨부 — pending draft에 저장됨
    const pendingInfo = readPendingExperimentInfo()
    const hasInfo = experimentInfoHasAny(pendingInfo)

    uploadAndAnalyze(file, {
      sampleFps: 1,
      info: hasInfo ? pendingInfo : undefined,
      source,
      onProgress: (pct) => setProgress(pct),
      onUploadComplete: () => {
        setProgress(100)
        navigate('/drafts', { replace: true })
      },
      onUploadError: (err) => {
        console.error('업로드 실패:', err)
        window.alert(`업로드 중 오류가 발생했습니다.\n${err}`)
        navigate(source === 'record' ? '/record' : '/upload', { replace: true })
      },
    })
  }, [file, navigate, source])

  const offset = CIRC - (Math.min(progress, 100) / 100) * CIRC

  const headerTitle = phase === 'upload' ? '영상 업로드' : '최종안 작성'
  const centerTitle = phase === 'upload' ? '영상 업로드' : '보고서 최종안'
  const centerSub = phase === 'upload' ? '업로드 중...' : '제작 중...'

  return (
    <AppShell variant="page">
      <PageHeader title={headerTitle} />
      <div className={styles.body}>
        <div
          className={styles.ringWrap}
          role="progressbar"
          aria-valuenow={Math.round(progress)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <svg className={styles.ring} viewBox="0 0 220 220" aria-hidden>
            {/* 배경 트랙 */}
            <circle cx="110" cy="110" r={R} fill="none" stroke="#d5d7e3" strokeWidth="20" />
            {/* 진행 호 */}
            <circle
              cx="110" cy="110" r={R}
              fill="none"
              stroke="var(--lab-lavender)"
              strokeWidth="20"
              strokeLinecap="round"
              strokeDasharray={`${CIRC} ${CIRC}`}
              strokeDashoffset={offset}
              className={styles.ringArc}
            />
          </svg>

          <div className={styles.ringCenter}>
            <img
              className={styles.centerIcon}
              src={lablogMark}
              alt=""
              width={840}
              height={814}
            />
            <p className={styles.centerTitle}>{centerTitle}</p>
            <p className={styles.centerSub}>{centerSub}</p>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
