import { useCallback, useState } from 'react'
import {
  clearPendingExperimentInfo,
  readPendingExperimentInfo,
  writePendingExperimentInfo,
  type ExperimentInfo,
} from '../api/lablog'
import styles from './ExperimentInfoForm.module.css'

type Props = {
  // 제어 모드 — value/onChange 둘 다 주어지면 localStorage 미사용.
  // AnalysisDetailPage에서 draft.info를 직접 편집할 때 사용.
  value?: ExperimentInfo
  onChange?: (next: ExperimentInfo) => void
  // 표시 제목 — 기본은 "기본 정보 입력 (선택)". 사후 편집 모드에서는 다른 카피 사용 가능.
  title?: string
  // 설명 문구 (헤더 아래 작은 텍스트)
  subtitle?: string
}

// UploadPage·RecordPage 둘 다 사용. localStorage에 자동 저장되어
// 페이지 이동 후 돌아와도 값이 유지된다. Draft 생성 시점에 그 값이 draft.info로 복사됨.
// AnalysisDetailPage에서는 value/onChange를 직접 받아 draft.info를 편집하는 controlled 모드로 동작.
// 초기에 입력된 게 있으면 펼친 상태로 시작 — 이어서 작성하기 편하도록.
export function ExperimentInfoForm({ value, onChange, title, subtitle }: Props = {}) {
  const controlled = value !== undefined && onChange !== undefined
  const [uncontrolledInfo, setUncontrolledInfo] = useState<ExperimentInfo>(
    () => (controlled ? {} : readPendingExperimentInfo()),
  )
  const info = controlled ? (value as ExperimentInfo) : uncontrolledInfo
  const initialHasAny = Object.values(info).some((v) => v && v.trim())
  const [open, setOpen] = useState(initialHasAny || controlled)

  const update = useCallback(
    (key: keyof ExperimentInfo) =>
      (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const next = { ...info, [key]: e.target.value }
        if (controlled) {
          onChange!(next)
        } else {
          setUncontrolledInfo(next)
          writePendingExperimentInfo(next)
        }
      },
    [controlled, info, onChange],
  )

  const clear = useCallback(() => {
    if (controlled) {
      onChange!({})
    } else {
      setUncontrolledInfo({})
      clearPendingExperimentInfo()
    }
  }, [controlled, onChange])

  return (
    <section className={styles.wrap}>
      <div
        className={styles.headerRow}
        onClick={() => setOpen((v) => !v)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            setOpen((v) => !v)
          }
        }}
        aria-expanded={open}
      >
        <h2 className={styles.title}>{title ?? '기본 정보 입력 (선택)'}</h2>
        <button
          type="button"
          className={styles.toggle}
          onClick={(e) => {
            e.stopPropagation()
            setOpen((v) => !v)
          }}
          aria-label={open ? '접기' : '펼치기'}
        >
          {open ? '▲ 접기' : '▼ 펼치기'}
        </button>
      </div>
      {open ? (
        <>
          <p className={styles.subtitle}>
            {subtitle ??
              '모든 항목은 선택사항입니다. 입력된 정보는 보고서 생성 시 함께 전달돼 더 정확한 결과를 만듭니다.'}
          </p>
          <div className={styles.fields}>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="info-title">실험 제목</label>
              <input
                id="info-title"
                className={styles.input}
                value={info.title ?? ''}
                onChange={update('title')}
                placeholder="예: 산-염기 중화반응 적정 실험"
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="info-subject">실험 주제</label>
              <input
                id="info-subject"
                className={styles.input}
                value={info.subject ?? ''}
                onChange={update('subject')}
                placeholder="예: 산과 염기의 중화점 측정"
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="info-date">실험 날짜</label>
              <input
                id="info-date"
                type="date"
                className={styles.input}
                value={info.date ?? ''}
                onChange={update('date')}
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="info-hypothesis">가설</label>
              <textarea
                id="info-hypothesis"
                className={styles.textarea}
                value={info.hypothesis ?? ''}
                onChange={update('hypothesis')}
                placeholder="예: HCl 농도를 정확히 알면 NaOH로 적정해 중화점을 찾을 수 있다."
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="info-other">기타</label>
              <textarea
                id="info-other"
                className={styles.textarea}
                value={info.other ?? ''}
                onChange={update('other')}
                placeholder="기타 참고사항, 변인 통제, 안전 주의사항 등"
              />
            </div>
          </div>
          <div className={styles.clearRow}>
            <button type="button" className={styles.clearBtn} onClick={clear}>
              모두 지우기
            </button>
          </div>
        </>
      ) : null}
    </section>
  )
}
