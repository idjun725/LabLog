import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AppShell } from '../components/AppShell'
import { ExperimentInfoForm } from '../components/ExperimentInfoForm'
import { PageHeader } from '../components/PageHeader'
import {
  WS_BASE,
  addDraft,
  readPendingExperimentInfo,
  synthesizeTTS,
  transcribeRecording,
  updateDraft,
  type AnalysisRecord,
  type TranscribeChunkMessage,
} from '../api/lablog'
import styles from './RecordPage.module.css'

const SEND_INTERVAL_MS = 1000  // 1초당 1프레임 — CPU 추론이라 더 빠르게 보내면 큐가 적체됨
const PREFERRED_CAMERA_KEY = 'lablog:preferredCameraId'
// 실시간 STT 청크 길이. MediaRecorder를 N초마다 재시작해 각 청크가 독립적인 WebM 파일이 되도록.
// (timeslice 옵션은 같은 컨테이너의 fragment를 만들어 Whisper가 단독 디코드 불가.)
// 6초로 설정 — Whisper는 30초 단위로 학습돼 청크가 길수록 정확도 ↑. 4초는 짧아 단어 잘림 심함.
// 8초 이상이면 지연이 체감되니 6초가 정확도-지연 균형점.
const REALTIME_STT_CHUNK_MS = 6000

// TTS 음성 안내 설정.
const TTS_ENABLED_KEY = 'lablog:ttsEnabled'
// 같은 문장은 이 간격 내 재생 안 함 — 깜빡이는 assist 메시지로 인한 음성 스팸 방지.
const TTS_MIN_INTERVAL_MS = 6000

function timestampToSec(ts: string): number {
  const parts = ts.split(':').map((p) => Number.parseInt(p, 10))
  if (parts.length !== 3 || parts.some(Number.isNaN)) return 0
  return parts[0] * 3600 + parts[1] * 60 + parts[2]
}

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

export function RecordPage() {
  const navigate = useNavigate()
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recordsRef = useRef<AnalysisRecord[]>([])
  const recordingStartRef = useRef<number>(0)
  // 마이크 녹음용 — 비디오와 별도 트랙으로 MediaRecorder를 돌린다.
  const audioRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  // 실시간 STT 전용 별도 MediaRecorder + WebSocket — N초마다 재시작해 독립 WebM 청크 생성.
  // 배치 audioRecorderRef와 동시 가동 (실시간 실패 시 fallback).
  const realtimeRecorderRef = useRef<MediaRecorder | null>(null)
  const transcribeWsRef = useRef<WebSocket | null>(null)
  const liveTranscriptsRef = useRef<{ elapsed_sec: number; text: string }[]>([])
  // TTS — assist 메시지를 음성으로 안내. 중복 재생 방지를 위해 last text/time 추적.
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null)
  const lastTtsTextRef = useRef<string>('')
  const lastTtsAtRef = useRef<number>(0)

  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [recording, setRecording] = useState(false)
  // 실시간 영상 보조 — 백엔드가 보내준 안내 메시지를 화면 상단에 띄운다.
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
  // 촬영 일시정지 — 프레임 전송·오디오 녹음·실시간 STT 모두 중단, 카메라 미리보기는 유지.
  // ref로 동기화해 interval/cycle 콜백에서 최신값 즉시 참조 가능.
  const [paused, setPaused] = useState(false)
  const pausedRef = useRef(false)
  // 일시정지 후 재개 시 다시 호출할 realtime STT cycle 시작 함수
  const startRealtimeCycleRef = useRef<(() => void) | null>(null)
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

  // 분석 모드: WebSocket을 통해 프레임을 백엔드로 전송하고 응답을 누적한다.
  // 변화가 있는 프레임(YOLO 객체 집합 변경 또는 새 OCR 텍스트)만 records에 저장한다.
  useEffect(() => {
    if (!recording || !stream) return

    recordsRef.current = []
    recordingStartRef.current = Date.now()
    let inflight = false
    let prevObjects = ''
    let prevOcr = ''
    let isFirst = true

    // ── 마이크 오디오 녹음 시작 (별도 MediaRecorder) ─────────────────
    audioChunksRef.current = []
    const audioTracks = stream.getAudioTracks()
    const audioMimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm'
    if (audioTracks.length > 0) {
      const audioOnly = new MediaStream(audioTracks)
      const recorder = new MediaRecorder(audioOnly, { mimeType: audioMimeType })
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }
      recorder.start()
      audioRecorderRef.current = recorder
    }

    // ── 실시간 STT — 별도 MediaRecorder 사이클 + WebSocket ───────────
    // 매 REALTIME_STT_CHUNK_MS마다 recorder를 stop()→재생성해 독립 WebM 청크 생성.
    // (timeslice는 같은 컨테이너의 fragment를 만들어 Whisper가 단독 디코드 불가.)
    liveTranscriptsRef.current = []
    let realtimeRestartTimer: number | null = null
    let realtimeActive = audioTracks.length > 0
    let transcribeWs: WebSocket | null = null

    if (realtimeActive) {
      transcribeWs = new WebSocket(`${WS_BASE}/ws/transcribe`)
      transcribeWs.binaryType = 'arraybuffer'
      transcribeWsRef.current = transcribeWs

      transcribeWs.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as TranscribeChunkMessage
          if (msg.text && !msg.skipped) {
            liveTranscriptsRef.current.push({
              elapsed_sec: msg.elapsed_sec,
              text: msg.text,
            })
          }
        } catch {
          /* parse error 무시 */
        }
      }
      transcribeWs.onerror = () => {
        console.warn('[transcribe-ws] 연결 오류 — 종료 후 배치 STT로 fallback됩니다.')
      }

      const startRealtimeCycle = () => {
        if (!realtimeActive || pausedRef.current) return
        const audioOnly = new MediaStream(audioTracks)
        const chunkParts: Blob[] = []
        const cycle = new MediaRecorder(audioOnly, { mimeType: audioMimeType })
        cycle.ondataavailable = (ev) => {
          if (ev.data.size > 0) chunkParts.push(ev.data)
        }
        cycle.onstop = async () => {
          if (chunkParts.length === 0) return
          // 일시정지 중 cycle stop으로 도착한 잔여 chunk는 보내지 않음 (불완전 데이터 우려)
          if (pausedRef.current) return
          const blob = new Blob(chunkParts, { type: audioMimeType })
          try {
            const buf = await blob.arrayBuffer()
            const wsRef = transcribeWsRef.current
            if (wsRef && wsRef.readyState === WebSocket.OPEN) {
              wsRef.send(buf)
            }
          } catch (err) {
            console.warn('[transcribe-ws] 청크 전송 실패:', err)
          }
        }
        cycle.start()
        realtimeRecorderRef.current = cycle

        realtimeRestartTimer = window.setTimeout(() => {
          try { cycle.stop() } catch { /* ignore */ }
          // 일시정지 중이면 재시작 안 함 — togglePause(resume)가 직접 호출
          if (!pausedRef.current) startRealtimeCycle()
        }, REALTIME_STT_CHUNK_MS)
      }
      // resume에서 다시 호출할 수 있도록 ref에 노출
      startRealtimeCycleRef.current = startRealtimeCycle
      startRealtimeCycle()
    }

    const ws = new WebSocket(`${WS_BASE}/ws/analyze`)
    ws.binaryType = 'arraybuffer'
    let intervalId: number | null = null

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
        const currentObjects = Object.keys(record.yolo).sort().join(',')
        const yoloChanged = currentObjects !== prevObjects
        const ocrChanged = !!record.ocr && record.ocr !== prevOcr

        if (isFirst || yoloChanged || ocrChanged) {
          recordsRef.current.push(record)
          prevObjects = currentObjects
          if (record.ocr) prevOcr = record.ocr
          isFirst = false
        }

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
      // 오디오 레코더가 아직 돌고 있으면 안전망으로 정리 (정상 흐름에선 handleToggle에서 이미 멈춰있음)
      const rec = audioRecorderRef.current
      if (rec && rec.state !== 'inactive') {
        try { rec.stop() } catch { /* ignore */ }
      }
      audioRecorderRef.current = null
      // 실시간 STT 정리 — 사이클 중단 → recorder stop → WS close
      realtimeActive = false
      if (realtimeRestartTimer !== null) {
        window.clearTimeout(realtimeRestartTimer)
        realtimeRestartTimer = null
      }
      const rtRec = realtimeRecorderRef.current
      if (rtRec && rtRec.state !== 'inactive') {
        try { rtRec.stop() } catch { /* ignore */ }
      }
      realtimeRecorderRef.current = null
      startRealtimeCycleRef.current = null
      // 다음 녹화 세션을 위해 일시정지 상태 초기화
      pausedRef.current = false
      setPaused(false)
      if (
        transcribeWs &&
        (transcribeWs.readyState === WebSocket.OPEN ||
          transcribeWs.readyState === WebSocket.CONNECTING)
      ) {
        transcribeWs.close()
      }
      transcribeWsRef.current = null
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
  const stopAudioRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = audioRecorderRef.current
      if (!recorder || recorder.state === 'inactive') {
        resolve(null)
        return
      }
      recorder.onstop = () => {
        const chunks = audioChunksRef.current
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
  //  - paused=true: 프레임 전송 stop (interval 가드), audio MediaRecorder.pause(),
  //    실시간 STT cycle 정지 (재시작 안 함)
  //  - paused=false: audio MediaRecorder.resume(), 실시간 STT cycle 재시작
  const togglePause = useCallback(() => {
    if (!recording) return
    setPaused((prev) => {
      const next = !prev
      pausedRef.current = next
      const audio = audioRecorderRef.current
      if (next) {
        // 일시정지
        if (audio && audio.state === 'recording') {
          try { audio.pause() } catch { /* ignore */ }
        }
        // 실시간 STT cycle 즉시 중단 (현재 진행 중인 chunk는 onstop에서 paused 체크로 폐기)
        const rtRec = realtimeRecorderRef.current
        if (rtRec && rtRec.state !== 'inactive') {
          try { rtRec.stop() } catch { /* ignore */ }
        }
        realtimeRecorderRef.current = null
      } else {
        // 재개
        if (audio && audio.state === 'paused') {
          try { audio.resume() } catch { /* ignore */ }
        }
        // 실시간 cycle 새로 시작
        startRealtimeCycleRef.current?.()
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
  //   중지 — 오디오 정리 → records 저장 → 실시간 transcripts로 speech 채움 → 부족 시 배치 STT fallback
  const handleToggleRecording = useCallback(async () => {
    if (recording) {
      // 1) 오디오 레코더 먼저 멈추고 최종 blob 받기 (effect cleanup이 돌기 전에)
      const audioBlob = await stopAudioRecording()
      // 1.5) 실시간 transcripts 스냅샷 — cleanup이 ref를 비우기 전에
      const transcripts = liveTranscriptsRef.current.slice()

      // 2) 녹화 상태 종료 (이때 effect cleanup이 WS close + 오디오 안전망 정리)
      setRecording(false)

      const rawRecords = recordsRef.current.slice()
      recordsRef.current = []
      if (rawRecords.length === 0) return

      // 3) 실시간 transcripts를 records.speech에 매핑 (elapsed_sec ~ timestamp 매칭)
      const sortedTr = [...transcripts].sort((a, b) => a.elapsed_sec - b.elapsed_sec)
      const records = rawRecords.map((r) => {
        const recSec = timestampToSec(r.timestamp)
        let matched = ''
        for (const t of sortedTr) {
          const tEnd = t.elapsed_sec + REALTIME_STT_CHUNK_MS / 1000
          if (t.elapsed_sec <= recSec && recSec <= tEnd) {
            matched = t.text
          }
        }
        return matched ? { ...r, speech: matched } : r
      })

      // 제목은 addDraft가 YYMMDD-N 형식으로 자동 생성. 사용자가 언제든 수정 가능.
      const info = readPendingExperimentInfo()
      const hasInfo = Object.values(info).some((v) => v && v.trim())
      const id = addDraft({
        source: 'record',
        data: { filename: null, frame_count: records.length, records },
        info: hasInfo ? info : undefined,
      })

      // 4) 실시간 transcripts가 비어있을 때만 배치 fallback 트리거
      const hasRealtime = transcripts.length > 0
      if (!hasRealtime && audioBlob && audioBlob.size > 0) {
        updateDraft(id, { audio_pending: true })
        transcribeRecording(id, audioBlob).catch((e: Error) => {
          console.error('[record] 오디오 전사 실패:', e)
          updateDraft(id, { audio_pending: false, error: e.message })
        })
      }

      // 5) 이동 (AnalysisDetailPage가 polling으로 STT 완료를 감지함)
      navigate(`/analysis/${id}`)
    } else {
      setRecording(true)
    }
  }, [recording, navigate, stopAudioRecording])

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
