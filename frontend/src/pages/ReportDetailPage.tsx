import { useCallback, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  deleteDraft,
  getDraft,
  hideMockReport,
  updateDraftTitle,
  type Draft,
  type GeneratedReport,
} from '../api/lablog'
import styles from './ReportDetailPage.module.css'

interface FinalReport {
  title: string
  소속: string
  학번: string
  이름: string
  실험날짜: string
  현미경번호: string
  실험제목: string
  실험목적: string[]
  준비물: string[]
  실험방법: string
  유의사항: string[]
}

const MOCK_REPORTS: Record<string, FinalReport> = {
  '1': {
    title: '보고서 최종안',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '',
    현미경번호: '',
    실험제목: '쥐 내부',
    실험목적: [
      '쥐의 주요 외부 특징 및 내부 장기를 확인한다.',
      '내부 장기들의 상대적인 위치를 서술한다.',
      '겸소를 찰거하여 실소 내 장기의 모양을 관찰한다.',
    ],
    준비물: ['쥐(암주쥐~수컷)', '해부대', '해부세트', '아하핵', '현미경'],
    실험방법: `1)마취
① 강추탈골
   침하(강기학과 침강공기학) 두개를 자루에 있는 독의 양쪽에 놓고 누른다. 다음 손으로는, 두개골로부터 강추를 분리시키기 위해 꼬리를 빠르게 잡아당긴다.
② 복강주사
   아하핵(클레핵) 주사체도 무방
③ 흡입마취
   주로 Chlorotom 사용, CO2 사용해도 무방

2)쥐의 복부가 위를 향하도록 발을 해부침시 위에 45도 각도로 단단히 핀을 꽂는다.

3)쥐의 꼬리가 자신의 반대 방향을 향하도록 높은 후, 흉부(가슴)의 밑바닥에 있는 녹골 주머니의 바로 아래를 가위로 약간 잘라주고 녹골주머니의 가장 윗쪽인 흠골이 잘릴 때까지 앞쪽으로 잘라낸다.

4)쥐를 돌려서 꼬리가 자신들을 향하도록 놓고 내장이 잘리지 않게 자르는 가위로 주머니로 소대를 향하여 복부를 잘라나간다. 잘라나간 다음에는 복부와 생식기관에 도달할 때까지 복부와 중앙 태쪽 표면 위의 칸선을 따라 절단한다.

5)생식기관에서 측면으로 비스듬히 거의 등뼈에 도달 할 때까지 자른다.

6)내부구조를 관찰한다. 그림과 대조하면서 여러 기관들을 찾아본다. 그리고 소화 기관, 호흡기관, 비뇨기관, 생식기관 등을 차례로 관찰한다.`,
    유의사항: [
      '처음 복부 자를 때 얇은 악 기리 칼을 집어 넣고 칼을 들어서 자른다.',
      '잡을 때 안기가 감싸고 있으므로, 그를 관찰하기 위해서이다.',
      '소감을 건널 때 끌거지지 않도록 한다.',
    ],
  },
  '2': {
    title: '보고서B',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-05-01',
    현미경번호: '',
    실험제목: '식물 잎의 기공 밀도와 광합성 효율',
    실험목적: ['식물 잎의 기공 밀도와 광합성 효율의 상관관계를 관찰한다.'],
    준비물: ['강낭콩', '시금치', '현미경', '표본 제작 도구', '광도계'],
    실험방법: `1)강낭콩과 시금치 잎의 표본을 제작한다.

2)현미경으로 각 잎의 기공 밀도를 측정한다. (단위: 개/mm²)

3)광도계를 이용해 각 식물의 광합성 효율을 측정한다.

4)기공 밀도와 광합성 효율의 상관관계를 분석한다.`,
    유의사항: [
      '표본 제작 시 잎이 찢어지지 않도록 주의한다.',
      '현미경 배율을 통일하여 비교한다.',
    ],
  },
  '3': {
    title: '보고서C',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-05-03',
    현미경번호: '',
    실험제목: '토양 pH에 따른 무 성장 속도 비교',
    실험목적: ['토양 pH에 따른 무 성장 속도 차이를 비교한다.'],
    준비물: ['무 씨앗 (동일 품종)', 'pH 조절 용액', '토양', '자', '재배 용기'],
    실험방법: `1)pH 5, 6, 7, 8 네 가지 환경의 토양을 준비한다.

2)각 환경에 동일 품종의 무를 심는다.

3)4주간 동일 조건(온도, 광량, 수분)으로 재배한다.

4)매주 성장 길이를 측정하고 기록한다.`,
    유의사항: [
      '온도와 수분 조건을 동일하게 유지한다.',
      '잎의 황변 여부도 함께 기록한다.',
    ],
  },
  '4': {
    title: '보고서D',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-04-20',
    현미경번호: '',
    실험제목: '물의 전기분해 시 전압에 따른 수소 발생량',
    실험목적: ['물의 전기분해 시 전압에 따른 수소 발생량 변화를 측정한다.'],
    준비물: ['전해질 수용액', '전극', '전원 장치', '수소 포집관', '메스실린더'],
    실험방법: `1)전해질 수용액에 전극을 삽입한다.

2)3V, 6V, 9V 전압을 각각 인가한다.

3)1분간 수소 포집량을 측정한다.

4)전압별 발생량을 비교·분석한다.`,
    유의사항: [
      '전극 삽입 시 합선에 주의한다.',
      '포집관의 기포를 완전히 제거한 후 측정을 시작한다.',
    ],
  },
  '5': {
    title: '보고서E',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-04-25',
    현미경번호: '',
    실험제목: '금속의 이온화 경향 비교',
    실험목적: ['다양한 금속의 이온화 경향을 실험으로 확인한다.'],
    준비물: ['Cu, Zn, Fe, Mg 금속 조각', '황산구리 수용액', '비커', '핀셋'],
    실험방법: `1)황산구리 수용액을 비커에 준비한다.

2)Cu, Zn, Fe, Mg 금속 조각을 각각 수용액에 담근다.

3)5분 후 반응 여부(색 변화, 석출 등)를 관찰한다.

4)이온화 경향 순서를 도출한다.`,
    유의사항: [
      '금속 표면의 산화막을 사포로 제거한 후 사용한다.',
      '관찰 후 폐액은 별도 수거한다.',
    ],
  },
  'archive-1': {
    title: '보고서1',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-03-10',
    현미경번호: '',
    실험제목: '삼투압 현상 관찰',
    실험목적: ['비커 내 용액의 농도 변화에 따른 삼투압 현상을 관찰한다.'],
    준비물: ['반투막', '설탕 용액', '증류수', '비커', '눈금자'],
    실험방법: `1)반투막에 고농도 설탕 용액을 넣어 봉한다.

2)증류수가 담긴 비커에 반투막을 넣는다.

3)30분 간격으로 수위 변화를 관찰·기록한다.`,
    유의사항: ['반투막에 구멍이 나지 않도록 주의한다.'],
  },
  'archive-2': {
    title: '보고서2',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-03-18',
    현미경번호: '',
    실험제목: '산-염기 지시약의 색 변화 범위 비교',
    실험목적: ['산-염기 지시약의 색 변화 범위를 비교한다.'],
    준비물: ['리트머스', '페놀프탈레인', 'BTB', '다양한 pH 표준액', '시험관'],
    실험방법: `1)각 지시약을 시험관에 분취한다.

2)다양한 pH의 표준액을 한 방울씩 첨가한다.

3)색 변화 구간을 기록한다.`,
    유의사항: ['지시약 첨가 시 정확한 양을 사용한다.'],
  },
  'archive-3': {
    title: '보고서3',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-03-25',
    현미경번호: '',
    실험제목: '빛의 파장에 따른 광합성 속도 비교',
    실험목적: ['빛의 파장(색)에 따른 광합성 속도 차이를 측정한다.'],
    준비물: ['수초(엘로데아)', '색 필터(청·녹·적)', '광량계', '산소 측정 장치'],
    실험방법: `1)수초를 물에 넣고 색 필터를 교체하며 빛을 조사한다.

2)각 파장별 산소 발생량을 측정한다.

3)파장과 광합성 속도의 관계를 분석한다.`,
    유의사항: ['광원과 수초의 거리를 동일하게 유지한다.'],
  },
  'archive-4': {
    title: '보고서4',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-04-02',
    현미경번호: '',
    실험제목: '효모의 발효 속도와 온도의 관계',
    실험목적: ['온도 변화에 따른 효모의 발효 속도를 측정한다.'],
    준비물: ['효모', '설탕 용액', '온도계', 'CO₂ 포집 장치', '수조'],
    실험방법: `1)설탕 용액에 효모를 첨가한다.

2)각 온도(20·35·45·60°C) 수조에 각각 보관한다.

3)10분 단위로 CO₂ 발생량을 측정한다.`,
    유의사항: ['온도 유지를 위해 뚜껑을 열지 않는다.'],
  },
  'archive-5': {
    title: '보고서5',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-04-10',
    현미경번호: '',
    실험제목: '자유 낙하 운동에서 중력 가속도 측정',
    실험목적: ['자유 낙하 실험을 통해 중력 가속도를 측정한다.'],
    준비물: ['쇠구슬', '낙하 측정 장치', '타이머', '자'],
    실험방법: `1)일정 높이에서 쇠구슬을 낙하시킨다.

2)낙하 시간을 측정한다.

3)h = ½gt² 식으로 g를 계산한다.

4)이론값(9.81 m/s²)과 비교한다.`,
    유의사항: ['바람이 없는 실내에서 실험한다.'],
  },
  'archive-6': {
    title: '보고서6',
    소속: '',
    학번: '',
    이름: '',
    실험날짜: '2026-04-18',
    현미경번호: '',
    실험제목: '단진자의 주기와 실의 길이 관계',
    실험목적: ['단진자의 주기와 실의 길이 관계를 실험으로 확인한다.'],
    준비물: ['추', '실', '각도기', '스톱워치', '스탠드'],
    실험방법: `1)실의 길이를 25, 50, 100, 200cm로 설정한다.

2)각 길이에서 10회 왕복 시간을 측정한다.

3)주기를 계산하고 T = 2π√(L/g)와 비교한다.`,
    유의사항: ['진폭을 10° 이하로 유지한다.'],
  },
}

function VerticalLabel({ text }: { text: string }) {
  return (
    <td className={styles.vLabel}>
      {text.split('').map((ch, i) => (
        <span key={i}>{ch}</span>
      ))}
    </td>
  )
}

export function ReportDetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()

  // draft에 Claude 생성 보고서가 있으면 그쪽을 우선 렌더링한다.
  const draft = getDraft(id)
  const generated = draft?.report

  const report = MOCK_REPORTS[id]
  // 실제 draft는 report 생성 후 보관함(/archive) 소속. mock은 id prefix로 분기.
  // archive-* mock → /archive, 그 외 mock(1-5) → /drafts.
  const backTo = draft
    ? '/archive'
    : id.startsWith('archive-')
      ? '/archive'
      : '/drafts'

  const handleDelete = useCallback(() => {
    if (!window.confirm('이 보고서를 삭제하시겠습니까?')) return
    if (draft) {
      deleteDraft(id)  // 실제 draft (Claude 생성 보고서 포함)
    } else {
      hideMockReport(id)  // mock 항목
    }
    navigate(backTo)
  }, [draft, id, backTo, navigate])

  if (generated && draft) {
    return (
      <GeneratedReportView
        draft={draft}
        report={generated}
        backTo={backTo}
        onDelete={handleDelete}
      />
    )
  }

  if (!report) {
    return (
      <div className={styles.shell}>
        <header className={styles.header}>
          <Link to={backTo} className={styles.back}>‹</Link>
          <h1 className={styles.title}>보고서</h1>
          <span className={styles.headerRight} />
        </header>
        <div className={styles.empty}>보고서를 찾을 수 없습니다.</div>
      </div>
    )
  }

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <Link to={backTo} className={styles.back}>‹</Link>
        <h1 className={styles.title}>{report.title}</h1>
        <button
          type="button"
          className={styles.deleteBtn}
          onClick={handleDelete}
          aria-label="삭제"
        >
          삭제
        </button>
        <button className={styles.downloadBtn} aria-label="다운로드">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </button>
      </header>

      <main className={styles.content}>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <tbody>
              <tr>
                <td className={styles.hLabel}>소 속</td>
                <td className={styles.hValue}>{report.소속}</td>
                <td className={styles.hLabel}>학번</td>
                <td className={styles.hValue}>{report.학번}</td>
                <td className={styles.hLabel}>이름</td>
                <td className={styles.hValue}>{report.이름}</td>
              </tr>
              <tr>
                <td className={styles.hLabel}>실험 날짜</td>
                <td colSpan={3} className={styles.hValue}>{report.실험날짜}</td>
                <td className={styles.hLabel}>현미경번호</td>
                <td className={styles.hValue}>{report.현미경번호}</td>
              </tr>
              <tr>
                <VerticalLabel text="실험제목" />
                <td colSpan={5} className={styles.sectionValue}>{report.실험제목}</td>
              </tr>
              <tr>
                <VerticalLabel text="실험목적" />
                <td colSpan={5} className={styles.sectionValue}>
                  {report.실험목적.map((p, i) => (
                    <div key={i} className={styles.bulletRow}>●{p}</div>
                  ))}
                </td>
              </tr>
              <tr>
                <VerticalLabel text="준비물" />
                <td colSpan={5} className={styles.sectionValue}>
                  {report.준비물.join(', ')}
                </td>
              </tr>
              <tr>
                <VerticalLabel text="실험방법" />
                <td colSpan={5} className={styles.sectionValue}>
                  <pre className={styles.pre}>{report.실험방법}</pre>
                </td>
              </tr>
              <tr>
                <VerticalLabel text="유의사항" />
                <td colSpan={5} className={styles.sectionValue}>
                  {report.유의사항.map((n, i) => (
                    <div key={i} className={styles.noteRow}>– {n}</div>
                  ))}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </main>
    </div>
  )
}

// Claude로 생성된 보고서 전용 뷰. mock 보고서 테이블 스타일을 재사용해 디자인을 통일.
function GeneratedReportView({
  draft,
  report,
  backTo,
  onDelete,
}: {
  draft: Draft
  report: GeneratedReport
  backTo: string
  onDelete: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [titleDraft, setTitleDraft] = useState(draft.title)

  const commitTitle = useCallback(() => {
    const trimmed = titleDraft.trim()
    setEditing(false)
    if (trimmed && trimmed !== draft.title) {
      updateDraftTitle(draft.id, trimmed)
    } else {
      setTitleDraft(draft.title)
    }
  }, [titleDraft, draft.id, draft.title])

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <Link to={backTo} className={styles.back}>‹</Link>
        {editing ? (
          <input
            className={styles.titleInput}
            value={titleDraft}
            autoFocus
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={commitTitle}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                commitTitle()
              } else if (e.key === 'Escape') {
                setEditing(false)
                setTitleDraft(draft.title)
              }
            }}
          />
        ) : (
          <button
            type="button"
            className={styles.titleEditable}
            onClick={() => {
              setTitleDraft(draft.title)
              setEditing(true)
            }}
            title="클릭하여 파일명 수정"
          >
            {draft.title} <span className={styles.titlePencil}>✎</span>
          </button>
        )}
        <Link
          to={`/analysis/${draft.id}`}
          className={styles.editAnalysisBtn}
          title="분석 결과 편집 / 보고서 재생성"
        >
          분석편집
        </Link>
        <button
          type="button"
          className={styles.deleteBtn}
          onClick={onDelete}
          aria-label="삭제"
        >
          삭제
        </button>
        <button className={styles.downloadBtn} aria-label="다운로드">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </button>
      </header>

      <main className={styles.content}>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <tbody>
              <tr>
                <td className={styles.hLabel}>실험 날짜</td>
                <td colSpan={5} className={styles.hValue}>{report.date || '데이터 부족'}</td>
              </tr>
              <tr>
                <VerticalLabel text="실험제목" />
                <td colSpan={5} className={styles.sectionValue}>{report.title}</td>
              </tr>
              <tr>
                <VerticalLabel text="선행연구" />
                <td colSpan={5} className={styles.sectionValue}>{report.preliminary_research}</td>
              </tr>
              <tr>
                <VerticalLabel text="실험목적" />
                <td colSpan={5} className={styles.sectionValue}>{report.objective}</td>
              </tr>
              <tr>
                <VerticalLabel text="가설" />
                <td colSpan={5} className={styles.sectionValue}>{report.hypothesis}</td>
              </tr>
              <tr>
                <VerticalLabel text="준비물" />
                <td colSpan={5} className={styles.sectionValue}>
                  {report.materials.length > 0 ? report.materials.join(', ') : '데이터 부족'}
                </td>
              </tr>
              <tr>
                <VerticalLabel text="실험방법" />
                <td colSpan={5} className={styles.sectionValue}>{report.method}</td>
              </tr>
              <tr>
                <VerticalLabel text="실험과정" />
                <td colSpan={5} className={styles.sectionValue}>
                  {report.procedure.map((p, i) => (
                    <div key={i} className={styles.bulletRow}>{i + 1}) {p}</div>
                  ))}
                </td>
              </tr>
              <tr>
                <VerticalLabel text="실험결과" />
                <td colSpan={5} className={styles.sectionValue}>
                  <pre className={styles.pre}>{report.results}</pre>
                </td>
              </tr>
              <tr>
                <VerticalLabel text="결론" />
                <td colSpan={5} className={styles.sectionValue}>
                  <pre className={styles.pre}>{report.conclusion}</pre>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </main>
    </div>
  )
}
