import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AppShell } from '../components/AppShell'
import { ExperimentInfoForm } from '../components/ExperimentInfoForm'
import { PageHeader } from '../components/PageHeader'
import {
  WS_BASE,
  synthesizeTTS,
  type AnalysisRecord,
} from '../api/lablog'
import styles from './RecordPage.module.css'

const SEND_INTERVAL_MS = 1000  // assist 배너용 실시간 프레임 전송 주기 (분석 결과엔 미반영)
const PREFERRED_CAMERA_KEY = 'lablog:preferredCameraId'

// TTS 음성 안내 설정.
const TTS_ENABLED_KEY = 'lablog:ttsEnabled'
// 같은 문장은 이 간격 내 재생 안 함 — 깜빡이는 assist 메시지로 인한 음성 스팸 방지.
const TTS_MIN_INTERVAL_MS = 6000

async function captureFrame(video: HTMLVideoElement): Promise<Blob | null> {
  if (video.readyState < 2 || !video.videoWidth) return null
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(video, 0, 0)
  return new Promise((resolve) =>
    canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.7),
  )
}

// 지원하는 것 중 가장 좋은 코덱 선택 — vp9가 없으면 vp8, 그것도 없으면 브라우저 기본.
function pickVideoMimeType(): string {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ]
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c
  }
  return 'video/webm'
}

export function RecordPage() {
  const navigate = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  // 촬영 전체를 하나의 영상 파일로 녹화 — 정지 후 업로드 분석(/api/analyze/video)과
  // 완전히 동일한 파이프라인(YOLO persistence filter, 배치 STT 등)으로 처리한다.
  // 이전에는 프레임 단위 실시간 분석 결과를 그대로 records로 썼는데, persistence
  // filter·배치 STT 문맥이 없어 정확도가 크게 떨어지는 문제가 있었다.
  const videoRecorderRef = useRef<MediaRecorder | null>(null)
  const videoChunksRef = useRef<Blob[]>([])
  // TTS — assist 메시지를 음성으로 안내. 중복 재생 방지를 위해 last text/time 추적.
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null)
  const lastTtsTextRef = useRef<string>('')
  const lastTtsAtRef = useRef<number>(0)

  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [recording, setRecording] = useState(false)
  // 실시간 영상 보조 — /ws/analyze가 매 프레임 계산해 보내주는 안내 메시지를 화면에
  // 띄우는 용도로만 쓴다 (최종 분석에는 반영되지 않는 순수 라이브 가이드).
  // 4초간 새 메시지가 안 오면 자동으로 사라진다 (조건이 해제된 것으로 간주).
  const [assistMessages, setAssistMessages] = useState<string[]>([])
  const assistTimerRef = useRef<number | null>(null)
  // 카메라 선택 — 시스템에 둘 이상의 카메라가 있을 때 드롭다운으로 전환 가능
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('')
  // TTS 음성 안내 on/off (localStorage 보존)
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(() => {
    return localStorage.getItem(TTS_ENABLED_KEY) !== 'false'
  })
  // 촬영 일시정지 — 영상 녹화·assist 프레임 전송 모두 중단, 카메라 미리보기는 유지.
  // ref로 동기화해 interval 콜백에서 최신값 즉시 참조 가능.
  const [paused, setPaused] = useState(false)
  const pausedRef = useRef(false)
  // 카메라 실제 비율 (videoWidth/videoHeight). 비디오 metadata 로드 시 갱신.
  // null이면 CSS 기본값(9:16)을 사용 — 모바일/세로 카메라 친화적.
  const [videoAspect, setVideoAspect] = useState<string | null>(null)

  // 카메라 권한 + 미리보기 — 페이지 진입 시 자동 켜짐, 이탈 시 자동 정리.
  // 이전 선택이 있으면 그 카메라를, 없으면 후면(또는 기본)을 시도.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const preferred = localStorage.getItem(PREFERRED_CAMERA_KEY)
        const videoConstraint: MediaTrackConstraints = preferred
          ? { deviceId: { ideal: preferred } }
          : { facingMode: { ideal: 'environment' } }
        const s = await navigator.mediaDevices.getUserMedia({
          video: videoConstraint,
          audio: true,
        })
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = s
        setStream(s)
        setError(null)
        const v = videoRef.current
        if (v) v.srcObject = s
      } catch {
        if (!cancelled) {
          setError('카메라·마이크 권한이 필요합니다. 브라우저 설정에서 허용해 주세요.')
        }
      }
    })()
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [])

  // 스트림이 잡힌 뒤 사용 가능한 카메라 목록 조회 + 현재 활성 deviceId 찾기.
  // (권한 부여 전에는 device.label이 빈 문자열이라 스트림 확보 후에 호출.)
  useEffect(() => {
    if (!stream) return
    let cancelled = false
    ;(async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        if (cancelled) return
        const videoInputs = devices.filter((d) => d.kind === 'videoinput')
        setCameras(videoInputs)
        const settings = stream.getVideoTracks()[0]?.getSettings()
        setSelectedDeviceId(settings?.deviceId ?? '')
      } catch {
        /* 열거 실패는 치명적 아님 — 단일 카메라처럼 동작 */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [stream])

  // 카메라 전환 — 촬영 중이면 무시. 새 deviceId로 getUserMedia 재호출.
  const switchCamera = useCallback(
    async (deviceId: string) => {
      if (recording) return
      if (!deviceId || deviceId === selectedDeviceId) return
      setError(null)
      try {
        // 이전 트랙 정리 (카메라 점유 해제)
        streamRef.current?.getTracks().forEach((t) => t.stop())
        const s = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: deviceId } },
          audio: true,
        })
        streamRef.current = s
        setStream(s)
        setSelectedDeviceId(deviceId)
        localStorage.setItem(PREFERRED_CAMERA_KEY, deviceId)
        const v = videoRef.current
        if (v) v.srcObject = s
      } catch (e) {
        setError(`카메라 전환 실패: ${(e as Error).message}`)
      }
    },
    [recording, selectedDeviceId],
  )

  useEffect(() => {
    const v = videoRef.current
    if (v && stream) v.srcObject = stream
  }, [stream])

  // 촬영 중 실행: (1) 스트림 전체를 하나의 webm 파일로 녹화 — 정지 시 업로드 분석과
  // 동일한 /api/analyze/video로 보낸다. (2) /ws/analyze로 프레임을 스트리밍해 assist
  // 배너·TTS만 갱신한다 (그 결과는 최종 분석에 쓰이지 않음 — 순수 라이브 가이드).
  useEffect(() => {
    if (!recording || !stream) return

    videoChunksRef.current = []
    const recorder = new MediaRecorder(stream, { mimeType: pickVideoMimeType() })
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) videoChunksRef.current.push(e.data)
    }
    recorder.start()
    videoRecorderRef.current = recorder

    const ws = new WebSocket(`${WS_BASE}/ws/analyze`)
    ws.binaryType = 'arraybuffer'
    let intervalId: number | null = null
    let inflight = false

    ws.onopen = () => {
      intervalId = window.setInterval(async () => {
        if (pausedRef.current) return  // 일시정지 중엔 프레임 전송 안 함
        const video = videoRef.current
        if (!video || ws.readyState !== WebSocket.OPEN) return
        if (inflight) return  // 이전 응답을 받기 전엔 다음 프레임을 보내지 않는다

        const blob = await captureFrame(video)
        if (!blob) return
        const buf = await blob.arrayBuffer()
        if (ws.readyState === WebSocket.OPEN) {
          inflight = true
          ws.send(buf)
        }
      }, SEND_INTERVAL_MS)
    }

    ws.onmessage = (e) => {
      inflight = false
      try {
        const record = JSON.parse(e.data) as AnalysisRecord
        // 실시간 영상 보조 — 매 프레임 안내 메시지 갱신.
        // 조건이 풀려도 4초 동안 메시지를 유지해 깜빡임 방지.
        if (record.assist && record.assist.length > 0) {
          const next = record.assist
          setAssistMessages((prev) => {
            // 메시지 배열이 동일하면 새 참조로 갱신하지 않는다
            // — 매 프레임 새 배열을 받아도 banner가 재렌더되지 않도록.
            if (
              prev.length === next.length &&
              prev.every((m, i) => m === next[i])
            ) {
              return prev
            }
            return next
          })
          if (assistTimerRef.current !== null) {
            window.clearTimeout(assistTimerRef.current)
          }
          assistTimerRef.current = window.setTimeout(() => {
            setAssistMessages([])
            assistTimerRef.current = null
          }, 4000)
        }
      } catch {
        /* parse error 무시 */
      }
    }

    ws.onerror = () => {
      console.error('WebSocket 오류 — 백엔드가 실행 중인지 확인하십시오.')
    }
    ws.onclose = () => {
      inflight = false
    }

    return () => {
      if (intervalId !== null) window.clearInterval(intervalId)
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
      // 녹화가 아직 돌고 있으면 안전망으로 정리 (정상 흐름에선 handleToggleRecording에서
      // 이미 stopVideoRecording을 통해 멈춰있음 — 언마운트 등 예외 경로 대비).
      const rec = videoRecorderRef.current
      if (rec && rec.state !== 'inactive') {
        try { rec.stop() } catch { /* ignore */ }
      }
      videoRecorderRef.current = null
      // 다음 녹화 세션을 위해 일시정지 상태 초기화
      pausedRef.current = false
      setPaused(false)
      // assist 타이머/메시지 정리
      if (assistTimerRef.current !== null) {
        window.clearTimeout(assistTimerRef.current)
        assistTimerRef.current = null
      }
      setAssistMessages([])
    }
  }, [recording, stream])

  // MediaRecorder를 중지하고 최종 blob을 반환 (onstop이 fire될 때까지 대기).
  // 트랙이 없거나 청크가 없으면 null.
  const stopVideoRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = videoRecorderRef.current
      if (!recorder || recorder.state === 'inactive') {
        resolve(null)
        return
      }
      recorder.onstop = () => {
        const chunks = videoChunksRef.current
        if (chunks.length === 0) {
          resolve(null)
          return
        }
        resolve(new Blob(chunks, { type: recorder.mimeType }))
      }
      recorder.stop()
    })
  }, [])

  // 일시정지 토글 — recording 중일 때만 의미 있음.
  // 영상 MediaRecorder의 네이티브 pause()/resume()만 다루면 된다 (오디오는 같은
  // 스트림의 트랙이라 함께 멈췄다 재개됨).
  const togglePause = useCallback(() => {
    if (!recording) return
    setPaused((prev) => {
      const next = !prev
      pausedRef.current = next
      const rec = videoRecorderRef.current
      if (next) {
        if (rec && rec.state === 'recording') {
          try { rec.pause() } catch { /* ignore */ }
        }
      } else {
        if (rec && rec.state === 'paused') {
          try { rec.resume() } catch { /* ignore */ }
        }
      }
      return next
    })
  }, [recording])

  // TTS 음성 안내 — assist 메시지가 변경될 때마다 mp3 받아 재생.
  // 중복/스팸 방지: 같은 텍스트는 TTS_MIN_INTERVAL_MS 안에 재생 안 함.
  useEffect(() => {
    if (!ttsEnabled || !recording) return
    if (assistMessages.length === 0) return
    const text = assistMessages.join(', ')
    if (text === lastTtsTextRef.current) return
    const now = Date.now()
    if (now - lastTtsAtRef.current < TTS_MIN_INTERVAL_MS) return
    // 이미 재생 중이면 새 문장은 스킵 — stale 안내 누적 방지
    const current = ttsAudioRef.current
    if (current && !current.paused && !current.ended) return

    lastTtsTextRef.current = text
    lastTtsAtRef.current = now

    let revoked = false
    ;(async () => {
      const blob = await synthesizeTTS(text)
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      ttsAudioRef.current = audio
      const cleanup = () => {
        if (revoked) return
        revoked = true
        URL.revokeObjectURL(url)
        if (ttsAudioRef.current === audio) ttsAudioRef.current = null
      }
      audio.onended = cleanup
      audio.onerror = cleanup
      audio.play().catch(() => {
        // autoplay 차단 또는 권한 문제 — 조용히 무시
        cleanup()
      })
    })()
  }, [assistMessages, ttsEnabled, recording])

  // TTS 토글 — localStorage 보존 + 현재 재생 즉시 중단
  const toggleTts = useCallback(() => {
    setTtsEnabled((prev) => {
      const next = !prev
      localStorage.setItem(TTS_ENABLED_KEY, String(next))
      if (!next) {
        const a = ttsAudioRef.current
        if (a && !a.paused) { try { a.pause() } catch { /* ignore */ } }
      }
      return next
    })
  }, [])

  // 촬영 토글:
  //   시작 — setRecording(true)
  //   중지 — 녹화 정지 → 녹화본을 LoadingPage로 넘겨 /api/analyze/video로 업로드
  //          (업로드 분석과 완전히 동일한 파이프라인 — persistence filter·배치 STT 포함)
  const handleToggleRecording = useCallback(async () => {
    if (recording) {
      // 1) 녹화 먼저 멈추고 최종 blob 받기 (effect cleanup이 돌기 전에)
      const videoBlob = await stopVideoRecording()

      // 2) 녹화 상태 종료 (이때 effect cleanup이 WS close + 안전망 정리)
      setRecording(false)

      if (!videoBlob || videoBlob.size === 0) return

      const file = new File(
        [videoBlob],
        `recording-${Date.now()}.webm`,
        { type: videoBlob.type || 'video/webm' },
      )

      // LoadingPage가 UploadPage와 완전히 같은 흐름(uploadAndAnalyze)으로 처리 —
      // 기본 정보(ExperimentInfoForm 입력값)는 공유 localStorage 키에서 자동으로 읽힌다.
      navigate('/loading', { state: { phase: 'upload', file, source: 'record' } })
    } else {
      setRecording(true)
    }
  }, [recording, navigate, stopVideoRecording])

  return (
    <AppShell variant="page">
      <PageHeader title="동영상 촬영" />

      <div
        className={styles.previewWrap}
        style={videoAspect ? { aspectRatio: videoAspect } : undefined}
      >
        <video
          ref={videoRef}
          className={styles.video}
          playsInline
          muted
          autoPlay
          onLoadedMetadata={(e) => {
            const v = e.currentTarget
            if (v.videoWidth && v.videoHeight) {
              setVideoAspect(`${v.videoWidth} / ${v.videoHeight}`)
            }
          }}
        />
        {!stream && !error ? (
          <div className={styles.placeholder}>카메라 준비 중…</div>
        ) : null}
        {paused ? (
          <div className={styles.pausedOverlay} role="status">⏸ 일시정지됨</div>
        ) : null}
        {error ? <div className={styles.error}>{error}</div> : null}
        {assistMessages.length > 0 ? (
          <div className={styles.assistBanner} role="alert">
            {assistMessages.map((m, i) => (
              <div key={i} className={styles.assistLine}>⚠️ {m}</div>
            ))}
          </div>
        ) : null}
      </div>

      <ExperimentInfoForm />

      <div className={styles.controls}>
        {cameras.length > 1 ? (
          <div className={styles.cameraRow}>
            <label htmlFor="camera-select" className={styles.cameraLabel}>
              카메라
            </label>
            <select
              id="camera-select"
              className={styles.cameraSelect}
              value={selectedDeviceId}
              onChange={(e) => switchCamera(e.target.value)}
              disabled={recording}
              title={recording ? '촬영 중에는 카메라를 변경할 수 없습니다' : ''}
            >
              {cameras.map((c) => (
                <option key={c.deviceId} value={c.deviceId}>
                  {c.label || `카메라 ${c.deviceId.slice(0, 8)}`}
                </option>
              ))}
            </select>
          </div>
        ) : null}

        <button
          type="button"
          className={`${styles.recordBtn} ${recording ? styles.recordBtnOn : ''}`}
          onClick={handleToggleRecording}
          disabled={!stream}
        >
          {recording ? '촬영 중지 (데모)' : '촬영 시작 (데모)'}
        </button>
        {recording ? (
          <button
            type="button"
            className={styles.secondary}
            onClick={togglePause}
          >
            {paused ? '▶ 재개' : '⏸ 일시정지'}
          </button>
        ) : null}
        <p className={styles.note}>
          실제 녹화·저장은 백엔드 연동 후 연결됩니다. 지금은 권한·미리보기만 확인할 수 있습니다. 개인정보가 수집될 수 있습니다.
        </p>
        <button
          type="button"
          className={styles.secondary}
          onClick={toggleTts}
          title={ttsEnabled ? '음성 안내를 끄면 assist 메시지가 화면에만 표시됩니다' : '음성 안내를 켜면 assist 메시지가 음성으로도 안내됩니다'}
        >
          {ttsEnabled ? '🔊 음성 안내 끄기' : '🔇 음성 안내 켜기'}
        </button>
        <Link to="/" className={styles.homeLink}>
          메인으로
        </Link>
      </div>
    </AppShell>
  )
}
