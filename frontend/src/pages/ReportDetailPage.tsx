import { useCallback } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ReportDetailView } from '../components/ReportDetailView'
import {
  deleteDraft,
  getDraft,
  hideMockReport,
  updateDraftTitle,
  type GeneratedReport,
} from '../api/lablog'
import styles from './ReportDetailPage.module.css'

// 카드 제목(드래프트/보관함 리스트에 보이는 라벨)과 본문 보고서를 함께 보관.
// cardTitle은 MOCK_ITEMS의 title과 일치시킨다.
const MOCK_REPORTS: Record<string, { cardTitle: string; report: GeneratedReport }> = {
  '1': {
    cardTitle: '보고서A',
    report: {
      title: '쥐 내부 해부',
      date: '2026-03-12',
      preliminary_research:
        '쥐는 척추동물의 대표적 실험 모델로 인간과 유사한 내부 장기 구조를 가져 해부 실습에 자주 활용된다. 소화·호흡·비뇨·생식 기관의 위치와 형태를 직접 관찰함으로써 척추동물 공통의 내부 구조를 이해할 수 있다.',
      objective:
        '쥐의 주요 외부 특징 및 내부 장기를 확인하고, 내부 장기들의 상대적인 위치를 서술한다. 횡격막을 절개하여 흉강 내 장기의 모양도 함께 관찰한다.',
      hypothesis:
        '쥐의 내부 장기 배치는 사람과 유사한 패턴을 보일 것이며, 소화·호흡·비뇨 기관이 흉강·복강에 명확히 구분되어 위치할 것이다.',
      materials: ['쥐(암컷·수컷)', '해부대', '해부세트', '아세트산', '현미경'],
      method:
        '쥐를 마취한 뒤 복부가 위를 향하도록 해부판에 고정하고, 흉강과 복강을 차례로 절개해 내부 장기를 정중선을 따라 관찰한다.',
      procedure: [
        '경추 탈골·복강주사·흡입마취 중 하나의 방법으로 쥐를 마취한다.',
        '쥐의 복부가 위를 향하도록 발을 해부판 위에 45° 각도로 단단히 고정한다.',
        '흉부 아래 늑골 주머니 바로 아래를 가위로 절개하고, 흉골이 잘릴 때까지 위로 잘라 올라간다.',
        '복부 중앙 정중선을 따라 생식기관에 도달할 때까지 절개한다.',
        '생식기관에서 측면으로 비스듬히 등뼈에 도달할 때까지 자른다.',
        '내부 구조를 관찰하며 소화 기관·호흡 기관·비뇨 기관·생식 기관을 차례로 확인한다.',
      ],
      results:
        '복강에서 간·위·소장·대장·신장이 명확히 구분되었고, 흉강에는 심장과 폐가 위치하며 횡격막이 두 강을 가르고 있었다. 사람의 장기 배치와 유사한 양상이 확인되었다.',
      conclusion:
        '쥐의 내부 장기 구조는 인간과 유사한 척추동물 공통 패턴을 따른다는 가설을 검증하였다. 해부 과정에서 횡격막의 역할과 흉강·복강의 구분이 분명히 관찰되었다.',
    },
  },
  '2': {
    cardTitle: '보고서B',
    report: {
      title: '식물 잎의 기공 밀도와 광합성 효율',
      date: '2026-05-01',
      preliminary_research:
        '기공은 식물 잎 표피에 분포한 가스 교환 통로로, 광합성에 필요한 CO₂ 흡수와 산소 방출, 증산 작용을 담당한다. 식물 종마다 기공 밀도가 달라 이것이 광합성 효율 차이의 한 원인일 수 있다.',
      objective: '식물 잎의 기공 밀도와 광합성 효율의 상관관계를 관찰한다.',
      hypothesis: '기공 밀도가 높을수록 단위 시간당 광합성 효율이 높을 것이다.',
      materials: ['강낭콩', '시금치', '현미경', '표본 제작 도구', '광도계'],
      method:
        '강낭콩과 시금치 잎의 표본을 제작해 현미경으로 기공 밀도를 측정하고, 광도계로 광합성 효율을 비교 분석한다.',
      procedure: [
        '강낭콩과 시금치 잎의 표본을 제작한다.',
        '현미경으로 각 잎의 기공 밀도를 측정한다 (단위: 개/mm²).',
        '광도계를 이용해 각 식물의 광합성 효율을 측정한다.',
        '기공 밀도와 광합성 효율의 상관관계를 분석한다.',
      ],
      results:
        '강낭콩은 약 180개/mm², 시금치는 약 120개/mm²의 기공 밀도를 보였으며, 광합성 효율도 강낭콩이 더 높게 측정되었다.',
      conclusion:
        '기공 밀도와 광합성 효율 사이에 양의 상관관계가 확인되었다. 다만 잎의 두께와 엽록체 수 등 다른 변수의 영향에 대해서는 추가 연구가 필요하다.',
    },
  },
  '3': {
    cardTitle: '보고서C',
    report: {
      title: '토양 pH에 따른 무 성장 속도 비교',
      date: '2026-05-03',
      preliminary_research:
        '식물의 뿌리는 토양의 pH에 따라 무기질 흡수 효율이 달라진다. 무는 약산성 토양(pH 6 부근)에서 잘 자라는 것으로 알려져 있으며, pH가 극단적으로 낮거나 높을수록 성장 저해가 관찰된다.',
      objective: '토양 pH에 따른 무 성장 속도 차이를 비교한다.',
      hypothesis: '약산성(pH 6) 토양에서 성장 속도가 가장 빠르고, 강산성·강염기 환경에서 둔화될 것이다.',
      materials: ['무 씨앗 (동일 품종)', 'pH 조절 용액', '토양', '자', '재배 용기'],
      method:
        'pH 5·6·7·8의 네 가지 토양을 준비해 동일 품종의 무를 4주간 같은 조건으로 재배하며 성장 길이를 매주 측정한다.',
      procedure: [
        'pH 5, 6, 7, 8 네 가지 환경의 토양을 준비한다.',
        '각 환경에 동일 품종의 무를 심는다.',
        '4주간 동일 조건(온도·광량·수분)으로 재배한다.',
        '매주 성장 길이를 측정하고 기록한다.',
      ],
      results:
        'pH 6 토양에서 가장 빠른 성장 속도(4주 평균 12cm)를 보였고, pH 5와 pH 8에서는 각각 8cm·9cm로 둔화되었다. pH 7은 11cm로 pH 6 다음으로 양호했다.',
      conclusion:
        '약산성 토양(pH 6)에서 무의 성장이 가장 활발했고, pH가 극단으로 갈수록 성장이 둔화된다는 가설을 지지하였다.',
    },
  },
  '4': {
    cardTitle: '보고서D',
    report: {
      title: '물의 전기분해 시 전압에 따른 수소 발생량',
      date: '2026-04-20',
      preliminary_research:
        '물의 전기분해는 전기 에너지를 이용해 수소와 산소를 발생시키는 산화·환원 반응이다. 인가 전압이 높을수록 전류량이 증가해 일정 시간 동안 발생하는 기체량도 증가할 것으로 예상된다.',
      objective: '물의 전기분해 시 전압에 따른 수소 발생량 변화를 측정한다.',
      hypothesis: '인가 전압이 클수록 1분당 수소 발생량이 비례하여 증가할 것이다.',
      materials: ['전해질 수용액', '전극', '전원 장치', '수소 포집관', '메스실린더'],
      method:
        '전해질 수용액에 전극을 담그고 3V·6V·9V를 차례로 인가한 뒤 1분간 발생하는 수소 기체의 양을 메스실린더로 측정한다.',
      procedure: [
        '전해질 수용액에 전극을 삽입한다.',
        '3V, 6V, 9V 전압을 각각 인가한다.',
        '1분간 수소 포집량을 측정한다.',
        '전압별 발생량을 비교·분석한다.',
      ],
      results:
        '3V에서 약 4mL, 6V에서 약 9mL, 9V에서 약 14mL의 수소가 발생해 전압 증가에 따라 발생량이 거의 선형으로 증가하였다.',
      conclusion:
        '전압이 증가할수록 수소 발생량도 비례하여 증가한다는 가설이 검증되었다. 다만 9V 부근에서는 전극 발열로 효율이 다소 감소하는 경향이 관찰되었다.',
    },
  },
  '5': {
    cardTitle: '보고서E',
    report: {
      title: '금속의 이온화 경향 비교',
      date: '2026-04-25',
      preliminary_research:
        '금속은 이온화 경향이 클수록 양이온이 되기 쉬워 다른 금속 이온과의 반응에서 환원제로 작용한다. 황산구리 수용액 속에서 더 활발한 금속은 구리를 석출시키는 반응을 보인다.',
      objective: '다양한 금속의 이온화 경향을 실험으로 확인한다.',
      hypothesis: 'Mg > Zn > Fe > Cu 순서로 이온화 경향이 크게 나타날 것이다.',
      materials: ['Cu, Zn, Fe, Mg 금속 조각', '황산구리 수용액', '비커', '핀셋'],
      method:
        '황산구리 수용액에 Cu·Zn·Fe·Mg 금속 조각을 각각 담가 5분간 색 변화와 석출 여부를 관찰하고 이온화 경향 순서를 도출한다.',
      procedure: [
        '황산구리 수용액을 비커에 준비한다.',
        'Cu, Zn, Fe, Mg 금속 조각을 각각 수용액에 담근다.',
        '5분 후 반응 여부(색 변화·석출 등)를 관찰한다.',
        '이온화 경향 순서를 도출한다.',
      ],
      results:
        'Mg는 격렬한 반응과 함께 다량의 구리가 빠르게 석출되었고, Zn·Fe 순으로 반응 강도가 약해졌다. Cu는 변화가 관찰되지 않았다.',
      conclusion:
        '관찰된 반응 강도 순서가 Mg > Zn > Fe > Cu로 가설과 일치하였으며, 이는 표준 환원 전위 표와도 부합한다.',
    },
  },
  'archive-1': {
    cardTitle: '보고서1',
    report: {
      title: '삼투압 현상 관찰',
      date: '2026-03-10',
      preliminary_research:
        '삼투압은 반투막을 사이에 두고 농도가 다른 용액이 접할 때 용매가 농도가 낮은 쪽에서 높은 쪽으로 이동하는 현상이다. 농도 차가 클수록 삼투 압력이 커진다.',
      objective: '비커 내 용액의 농도 변화에 따른 삼투압 현상을 관찰한다.',
      hypothesis: '반투막 내부 용액의 농도가 높을수록 외부 증류수가 더 많이 유입되어 수위가 빠르게 상승할 것이다.',
      materials: ['반투막', '설탕 용액', '증류수', '비커', '눈금자'],
      method:
        '반투막에 고농도 설탕 용액을 봉입한 뒤 증류수가 담긴 비커에 담그고 일정 시간 간격으로 수위 변화를 측정한다.',
      procedure: [
        '반투막에 고농도 설탕 용액을 넣어 봉한다.',
        '증류수가 담긴 비커에 반투막을 넣는다.',
        '30분 간격으로 수위 변화를 관찰·기록한다.',
      ],
      results: '30분마다 반투막 내부 수위가 평균 0.8cm씩 상승했고, 2시간 후 약 3.2cm가 증가하였다.',
      conclusion:
        '농도가 높은 쪽으로 용매가 이동한다는 삼투압 원리가 확인되었다. 시간 경과에 따른 수위 증가율은 점차 둔화되었으며 이는 농도 차가 줄어들기 때문으로 해석된다.',
    },
  },
  'archive-2': {
    cardTitle: '보고서2',
    report: {
      title: '산-염기 지시약의 색 변화 범위 비교',
      date: '2026-03-18',
      preliminary_research:
        '산-염기 지시약은 pH 변화에 따라 색이 변하는 화합물로, 각 지시약마다 색 변화가 일어나는 고유의 pH 범위가 있다. 적정 실험에서 적절한 지시약 선택의 기준이 된다.',
      objective: '산-염기 지시약의 색 변화 범위를 비교한다.',
      hypothesis: '리트머스·페놀프탈레인·BTB는 각각 다른 pH 영역에서 색 변화를 보일 것이다.',
      materials: ['리트머스', '페놀프탈레인', 'BTB', '다양한 pH 표준액', '시험관'],
      method:
        '각 지시약을 시험관에 분취하고 pH가 다른 표준액을 한 방울씩 떨어뜨려 색이 변하는 pH 구간을 관찰한다.',
      procedure: [
        '각 지시약을 시험관에 분취한다.',
        '다양한 pH의 표준액을 한 방울씩 첨가한다.',
        '색 변화 구간을 기록한다.',
      ],
      results:
        '리트머스는 pH 4.5–8.3, 페놀프탈레인은 pH 8.2–10.0(무색→분홍), BTB는 pH 6.0–7.6(노랑→파랑) 구간에서 색 변화가 관찰되었다.',
      conclusion:
        '지시약마다 색 변화 범위가 명확히 달라, 측정하려는 pH 영역에 맞는 지시약을 선택해야 함을 확인하였다.',
    },
  },
  'archive-3': {
    cardTitle: '보고서3',
    report: {
      title: '빛의 파장에 따른 광합성 속도 비교',
      date: '2026-03-25',
      preliminary_research:
        '엽록소는 청색광(약 430nm)과 적색광(약 660nm)에서 흡수율이 가장 높고 녹색광에서는 흡수율이 낮다. 따라서 광합성 속도도 파장에 따라 차이를 보일 것으로 예상된다.',
      objective: '빛의 파장(색)에 따른 광합성 속도 차이를 측정한다.',
      hypothesis: '청색·적색 파장에서 광합성 속도가 가장 크고, 녹색 파장에서 가장 작을 것이다.',
      materials: ['수초(엘로데아)', '색 필터(청·녹·적)', '광량계', '산소 측정 장치'],
      method:
        '수초에 색 필터를 통과한 빛을 조사하며 각 파장별로 발생하는 산소량을 측정해 광합성 속도를 비교한다.',
      procedure: [
        '수초를 물에 넣고 색 필터를 교체하며 빛을 조사한다.',
        '각 파장별 산소 발생량을 측정한다.',
        '파장과 광합성 속도의 관계를 분석한다.',
      ],
      results:
        '적색광에서 1분당 18기포, 청색광에서 16기포, 녹색광에서 5기포가 관찰되어 적색 > 청색 ≫ 녹색 순으로 광합성 속도가 측정되었다.',
      conclusion:
        '엽록소의 흡수 스펙트럼과 일치하는 결과가 나왔다. 녹색광에서 광합성이 거의 일어나지 않는다는 점이 확인되었다.',
    },
  },
  'archive-4': {
    cardTitle: '보고서4',
    report: {
      title: '효모의 발효 속도와 온도의 관계',
      date: '2026-04-02',
      preliminary_research:
        '효모는 무산소 조건에서 당을 분해해 에탄올과 CO₂를 생성하는 발효를 수행한다. 효모 효소는 약 35–40°C에서 최적 활성을 가지며 그 이상에서는 변성으로 활성이 급감한다.',
      objective: '온도 변화에 따른 효모의 발효 속도를 측정한다.',
      hypothesis: '35°C 부근에서 발효 속도가 가장 빠르고, 60°C 이상에서는 효소 변성으로 발효가 급격히 둔화될 것이다.',
      materials: ['효모', '설탕 용액', '온도계', 'CO₂ 포집 장치', '수조'],
      method:
        '설탕 용액에 효모를 첨가한 뒤 20·35·45·60°C 수조에서 10분 단위로 CO₂ 발생량을 측정한다.',
      procedure: [
        '설탕 용액에 효모를 첨가한다.',
        '각 온도(20·35·45·60°C) 수조에 각각 보관한다.',
        '10분 단위로 CO₂ 발생량을 측정한다.',
      ],
      results:
        '35°C에서 10분당 평균 26mL의 CO₂가 발생해 가장 활발했고, 45°C(18mL) > 20°C(9mL) > 60°C(2mL) 순이었다.',
      conclusion:
        '효모 효소의 최적 활성 온도가 35°C 부근임이 확인되었고, 60°C 이상에서는 단백질 변성으로 발효가 거의 멈추는 양상이 관찰되었다.',
    },
  },
  'archive-5': {
    cardTitle: '보고서5',
    report: {
      title: '자유 낙하 운동에서 중력 가속도 측정',
      date: '2026-04-10',
      preliminary_research:
        '공기 저항이 무시될 때 자유 낙하하는 물체의 운동은 등가속도 운동이며, h = ½gt² 식으로 낙하 시간과 높이로부터 중력 가속도 g를 산출할 수 있다.',
      objective: '자유 낙하 실험을 통해 중력 가속도를 측정한다.',
      hypothesis: '측정값이 이론값(9.81 m/s²)과 ±3% 이내로 일치할 것이다.',
      materials: ['쇠구슬', '낙하 측정 장치', '타이머', '자'],
      method:
        '일정 높이에서 쇠구슬을 자유 낙하시키고 낙하 시간을 측정한 뒤 h = ½gt² 식으로 g를 계산한다.',
      procedure: [
        '일정 높이에서 쇠구슬을 낙하시킨다.',
        '낙하 시간을 측정한다.',
        'h = ½gt² 식으로 g를 계산한다.',
        '이론값(9.81 m/s²)과 비교한다.',
      ],
      results:
        '높이 1.5m에서 평균 낙하 시간 0.554초가 측정되어 g ≈ 9.78 m/s²로 산출되었다. 이론값 대비 오차 0.3%.',
      conclusion:
        '측정 결과가 이론값과 ±1% 이내로 일치하여 가설을 만족하였다. 약간의 공기 저항과 타이머 반응 시간이 미세 오차의 원인으로 추정된다.',
    },
  },
  'archive-6': {
    cardTitle: '보고서6',
    report: {
      title: '단진자의 주기와 실의 길이 관계',
      date: '2026-04-18',
      preliminary_research:
        '단진자의 주기 T는 진폭이 작을 때 T = 2π√(L/g)로 근사된다. 즉 주기는 실의 길이의 제곱근에 비례하며 중력 가속도에는 반비례한다.',
      objective: '단진자의 주기와 실의 길이 관계를 실험으로 확인한다.',
      hypothesis: '주기는 실의 길이의 제곱근에 비례할 것이다.',
      materials: ['추', '실', '각도기', '스톱워치', '스탠드'],
      method:
        '실의 길이를 변화시키며 각 길이에서 단진자의 10회 왕복 시간을 측정해 주기를 산출하고 이론식과 비교한다.',
      procedure: [
        '실의 길이를 25·50·100·200cm로 설정한다.',
        '각 길이에서 10회 왕복 시간을 측정한다.',
        '주기를 계산하고 T = 2π√(L/g)와 비교한다.',
      ],
      results:
        '측정 주기는 25cm에서 1.00초, 50cm에서 1.42초, 100cm에서 2.00초, 200cm에서 2.83초로 측정되었다. 모두 이론값과 1% 이내로 일치.',
      conclusion:
        '주기가 실의 길이의 제곱근에 비례한다는 가설이 확인되었으며, T = 2π√(L/g) 식의 타당성이 검증되었다.',
    },
  },
}

export function ReportDetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()

  const draft = getDraft(id)
  const draftReport = draft?.report
  const mockEntry = MOCK_REPORTS[id]

  const backTo = draft
    ? '/archive'
    : id.startsWith('archive-')
      ? '/archive'
      : '/drafts'

  const handleDelete = useCallback(() => {
    if (!window.confirm('이 보고서를 삭제하시겠습니까?')) return
    if (draft) {
      deleteDraft(id)
    } else {
      hideMockReport(id)
    }
    navigate(backTo)
  }, [draft, id, backTo, navigate])

  if (draftReport && draft) {
    return (
      <ReportDetailView
        report={draftReport}
        cardTitle={draft.title}
        backTo={backTo}
        onDelete={handleDelete}
        onTitleChange={(t) => updateDraftTitle(draft.id, t)}
        analysisEditHref={`/analysis/${draft.id}`}
      />
    )
  }

  if (mockEntry) {
    return (
      <ReportDetailView
        report={mockEntry.report}
        cardTitle={mockEntry.cardTitle}
        backTo={backTo}
        onDelete={handleDelete}
      />
    )
  }

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
