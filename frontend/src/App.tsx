import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AnalysisDetailPage } from './pages/AnalysisDetailPage'
import { ArchivePage } from './pages/ArchivePage'
import { DraftsPage } from './pages/DraftsPage'
import { HomePage } from './pages/HomePage'
import { LoadingPage } from './pages/LoadingPage'
import { RecordPage } from './pages/RecordPage'
import { ReportDetailPage } from './pages/ReportDetailPage'
import { UploadPage } from './pages/UploadPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/record" element={<RecordPage />} />
        <Route path="/archive" element={<ArchivePage />} />
        <Route path="/drafts" element={<DraftsPage />} />
        <Route path="/loading" element={<LoadingPage />} />
        <Route path="/report/:id" element={<ReportDetailPage />} />
        <Route path="/analysis/:id" element={<AnalysisDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
