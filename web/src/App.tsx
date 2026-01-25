import { useState } from 'react'
import { MessageSquare, Settings, Database, BarChart3, Menu, X } from 'lucide-react'
import ChatInterface from './components/ChatInterface'
import TrainingInterface from './components/TrainingInterface'
import ModelsInterface from './components/ModelsInterface'
import AnalyticsInterface from './components/AnalyticsInterface'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [menuOpen, setMenuOpen] = useState(false)

  const tabs = [
    { id: 'chat', name: 'Chat', icon: MessageSquare },
    { id: 'training', name: 'Training', icon: Settings },
    { id: 'models', name: 'Models', icon: Database },
    { id: 'analytics', name: 'Analytics', icon: BarChart3 },
  ]

  return (
    <div style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', backgroundColor: '#020617', color: '#f1f5f9', overflow: 'hidden' }}>
      {/* Header */}
      <header style={{ height: '64px', padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', backgroundColor: '#0f172a', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            style={{ padding: '10px', background: 'transparent', border: 'none', color: '#f1f5f9', cursor: 'pointer', borderRadius: '8px' }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1e293b'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
          >
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
          <h2 style={{ fontSize: '18px', fontWeight: 600 }}>
            {tabs.find(t => t.id === activeTab)?.name}
          </h2>
        </div>
        <div style={{ fontSize: '13px', color: '#64748b' }}>
          VoxelLM v2.0
        </div>
      </header>

      <div style={{ flex: 1, display: 'flex', position: 'relative', overflow: 'hidden' }}>
        {/* Sidebar */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          height: '100%',
          width: '260px',
          backgroundColor: '#0f172a',
          borderRight: '1px solid #1e293b',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 10,
          transform: menuOpen ? 'translateX(0)' : 'translateX(-100%)',
          transition: 'transform 0.2s'
        }}>
          {/* Logo */}
          <div style={{ height: '64px', padding: '0 20px', display: 'flex', alignItems: 'center', borderBottom: '1px solid #1e293b' }}>
            <h1 style={{ fontSize: '18px', fontWeight: 600 }}>
              <span style={{ color: '#3b82f6' }}>Voxel</span>LM
            </h1>
          </div>

          {/* Navigation */}
          <nav style={{ flex: 1, padding: '16px' }}>
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id)
                    setMenuOpen(false)
                  }}
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    borderRadius: '10px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '6px',
                    fontSize: '15px',
                    fontWeight: 500,
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: isActive ? '#2563eb' : 'transparent',
                    color: isActive ? '#ffffff' : '#94a3b8',
                    transition: 'all 0.15s'
                  }}
                  onMouseEnter={(e) => !isActive && (e.currentTarget.style.backgroundColor = '#1e293b', e.currentTarget.style.color = '#e2e8f0')}
                  onMouseLeave={(e) => !isActive && (e.currentTarget.style.backgroundColor = 'transparent', e.currentTarget.style.color = '#94a3b8')}
                >
                  <Icon size={18} />
                  {tab.name}
                </button>
              )
            })}
          </nav>

          {/* Footer */}
          <div style={{ padding: '16px', borderTop: '1px solid #1e293b' }}>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <div style={{ width: '6px', height: '6px', backgroundColor: '#22c55e', borderRadius: '50%' }}></div>
                <span>API Connected</span>
              </div>
              <div>localhost:8000</div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {activeTab === 'chat' && <ChatInterface />}
          {activeTab === 'training' && <TrainingInterface />}
          {activeTab === 'models' && <ModelsInterface />}
          {activeTab === 'analytics' && <AnalyticsInterface />}
        </div>
      </div>
    </div>
  )
}

export default App
