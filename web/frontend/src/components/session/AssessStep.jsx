import { useState }          from 'react'
import { Card, TabBar }      from '../shared'
import RecordTab             from '../input/RecordTab'
import UploadTab             from '../input/UploadTab'
import TypeTab               from '../input/TypeTab'

const TABS = [
  { id: 'record', label: '🎙 Record Voice' },
  { id: 'upload', label: '📂 Upload File'  },
  { id: 'type',   label: '⌨️  Type Text'  },
]

export default function AssessStep({ loading, onText, onFile, onAudio }) {
  const [tab, setTab] = useState('record')
  return (
    <Card title="Speech Assessment" subtitle="Let us understand how you are speaking today.">
      <TabBar tabs={TABS} active={tab} onChange={setTab} />
      {tab === 'record' && <RecordTab loading={loading} onAudio={onAudio} />}
      {tab === 'upload' && <UploadTab loading={loading} onFile={onFile}   />}
      {tab === 'type'   && <TypeTab   loading={loading} onText={onText}   />}
    </Card>
  )
}
