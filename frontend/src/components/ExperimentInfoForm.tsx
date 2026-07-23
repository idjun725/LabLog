import { useCallback, useState } from 'react'
import {
  clearPendingExperimentInfo,
  experimentInfoHasAny,
  readPendingExperimentInfo,
  writePendingExperimentInfo,
  type ExperimentInfo,
} from '../api/lablog'
import styles from './ExperimentInfoForm.module.css'

// best.pt가 인식 가능한 25개 고정 클래스. en 값은 vectorizer.YOLO_VOCAB과 순서·값이
// 정확히 일치해야 한다(백엔드가 이 문자열로 클래스 인덱스를 찾음). 기본은 전부 인식되고,
// 사용자가 일부를 선택하면 그 클래스들로만 탐지가 제한된다(allowedClasses).
const AUTO_DETECTED_CLASSES: { ko: string; en: string }[] = [
  { ko: '비커', en: 'Beaker' },
  { ko: '부흐너 깔때기', en: 'Buchner_Funnel' },
  { ko: '뷰렛 스탠드', en: 'Burette_Stands' },
  { ko: '열량계', en: 'Calorimeter' },
  { ko: '삼각 플라스크', en: 'Conical_Flask' },
  { ko: '깔때기', en: 'Funnel' },
  { ko: '유리 막대', en: 'Glass_Rod' },
  { ko: '메스실린더', en: 'Measuring_Cylinder' },
  { ko: '기계식 저울', en: 'Mechanical_Balance_Scale' },
  { ko: '네슬러관(비색관)', en: 'Nessler_Reagent_Bottle' },
  { ko: '피펫', en: 'Pipette' },
  { ko: '유발과 유봉', en: 'Porcelain_Mortar Pestle' },
  { ko: '정밀 전자저울', en: 'Precision_Weight_Scale' },
  { ko: '시약병', en: 'Reagent_Bottle' },
  { ko: '둥근바닥 플라스크 (구경 1개)', en: 'Round_Bottom_Flask_Borosilicate_Glass_1_Neck' },
  { ko: '둥근바닥 플라스크 (구경 2개)', en: 'Round_Bottom_Flask_Borosilicate_Glass_2_Neck' },
  { ko: '둥근바닥 플라스크 (구경 3개)', en: 'Round_Bottom_Flask_Borosilicate_Glass_3_Neck' },
  { ko: '분액 깔때기', en: 'Separating_Funnel' },
  { ko: '알코올램프', en: 'Spirit_Lamp' },
  { ko: '시험관 집게', en: 'TestTube_Holder' },
  { ko: '시험관', en: 'Test_Tube' },
  { ko: '부피 플라스크', en: 'Volumetric_Flask' },
  { ko: '부피 피펫', en: 'Volumetric_Pipet' },
  { ko: '세척병', en: 'Wash_Bottle' },
  { ko: '칭량병', en: 'Weighing_Bottle' },
]

type Props = {
  // 제어 모드 — value/onChange 둘 다 주어지면 localStorage 미사용.
  // AnalysisDetailPage에서 draft.info를 직접 편집할 때 사용.
  value?: ExperimentInfo
  onChange?: (next: ExperimentInfo) => void
  // 표시 제목 — 기본은 "기본 정보 입력 (선택)". 사후 편집 모드에서는 다른 카피 사용 가능.
  title?: string
  // 설명 문구 (헤더 아래 작은 텍스트)
  subtitle?: string
  // 탐지 대상 객체(YOLO-World 프롬프트) 칩 선택 섹션 표시 여부.
  // 분석 전(UploadPage)에서만 의미 있음 — 촬영 중(RecordPage)엔 /ws/analyze가 아직
  // custom_classes를 지원하지 않고, 분석 후 편집(AnalysisDetailPage)은 이미 끝난
  // 탐지에 영향을 줄 수 없으므로 기본값 false.
  showClassSelector?: boolean
}

// UploadPage·RecordPage 둘 다 사용. localStorage에 자동 저장되어
// 페이지 이동 후 돌아와도 값이 유지된다. Draft 생성 시점에 그 값이 draft.info로 복사됨.
// AnalysisDetailPage에서는 value/onChange를 직접 받아 draft.info를 편집하는 controlled 모드로 동작.
// 초기에 입력된 게 있으면 펼친 상태로 시작 — 이어서 작성하기 편하도록.
export function ExperimentInfoForm({
  value,
  onChange,
  title,
  subtitle,
  showClassSelector = false,
}: Props = {}) {
  const controlled = value !== undefined && onChange !== undefined
  const [uncontrolledInfo, setUncontrolledInfo] = useState<ExperimentInfo>(
    () => (controlled ? {} : readPendingExperimentInfo()),
  )
  const info = controlled ? (value as ExperimentInfo) : uncontrolledInfo
  const initialHasAny = experimentInfoHasAny(info)
  const [open, setOpen] = useState(initialHasAny || controlled)
  const [customInput, setCustomInput] = useState('')

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

  const setCustomClasses = useCallback(
    (nextClasses: string[]) => {
      const next = { ...info, customClasses: nextClasses.length > 0 ? nextClasses : undefined }
      if (controlled) {
        onChange!(next)
      } else {
        setUncontrolledInfo(next)
        writePendingExperimentInfo(next)
      }
    },
    [controlled, info, onChange],
  )

  const toggleAllowedClass = useCallback(
    (en: string) => {
      const current = info.allowedClasses ?? []
      const nextClasses = current.includes(en)
        ? current.filter((c) => c !== en)
        : [...current, en]
      const next = {
        ...info,
        allowedClasses: nextClasses.length > 0 ? nextClasses : undefined,
      }
      if (controlled) {
        onChange!(next)
      } else {
        setUncontrolledInfo(next)
        writePendingExperimentInfo(next)
      }
    },
    [controlled, info, onChange],
  )

  const addCustomClass = useCallback(() => {
    const term = customInput.trim()
    if (!term) return
    const current = info.customClasses ?? []
    if (!current.includes(term)) {
      setCustomClasses([...current, term])
    }
    setCustomInput('')
  }, [customInput, info.customClasses, setCustomClasses])

  const removeClass = useCallback(
    (term: string) => {
      setCustomClasses((info.customClasses ?? []).filter((c) => c !== term))
    },
    [info.customClasses, setCustomClasses],
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
            {showClassSelector ? (
              <div className={styles.field}>
                <label className={styles.label}>탐지 대상 객체 (선택)</label>
                <p className={styles.classHint}>
                  아래 {AUTO_DETECTED_CLASSES.length}개는 기본 인식 목록입니다. 아무것도
                  선택하지 않으면 25개 전부를 인식합니다. 영상에 나오는 것만 선택하면
                  그 항목들로만 탐지를 제한해 다른 기자재와 헷갈리는 것을 줄입니다.
                </p>
                <div className={styles.chipRow}>
                  {AUTO_DETECTED_CLASSES.map((c) => {
                    const active = (info.allowedClasses ?? []).includes(c.en)
                    return (
                      <button
                        key={c.en}
                        type="button"
                        className={`${styles.chip} ${active ? styles.chipActive : ''}`}
                        aria-pressed={active}
                        onClick={() => toggleAllowedClass(c.en)}
                      >
                        {c.ko}
                      </button>
                    )
                  })}
                </div>
                <p className={styles.classHint}>목록에 없는 사물은 직접 추가해주세요.</p>
                <div className={styles.customAddRow}>
                  <input
                    className={styles.input}
                    value={customInput}
                    onChange={(e) => setCustomInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        addCustomClass()
                      }
                    }}
                    placeholder="목록에 없는 사물 직접 추가 (예: 자석)"
                  />
                  <button type="button" className={styles.addBtn} onClick={addCustomClass}>
                    추가
                  </button>
                </div>
                {(info.customClasses ?? []).length > 0 ? (
                  <div className={styles.chipRow}>
                    {(info.customClasses ?? []).map((term) => (
                      <button
                        key={term}
                        type="button"
                        className={`${styles.chip} ${styles.chipActive}`}
                        onClick={() => removeClass(term)}
                        title="탭하여 제거"
                      >
                        {term} ✕
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
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
