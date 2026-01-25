import { useState, useEffect } from 'react';
import { TrendingUp, Cpu, Database as DatabaseIcon, Zap } from 'lucide-react';
import { apiClient } from '../lib/api-client';
import type { SystemStatus } from '../types/api';

export default function AnalyticsInterface() {
    const [status, setStatus] = useState<SystemStatus | null>(null);

    useEffect(() => {
        const loadStatus = async () => {
            try {
                const data = await apiClient.getStatus();
                setStatus(data);
            } catch (error) {
                console.error('Failed to load status:', error);
            }
        };

        loadStatus();
        const interval = setInterval(loadStatus, 2000);
        return () => clearInterval(interval);
    }, []);

    const formatUptime = (seconds: number) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours}h ${minutes}m ${secs}s`;
    };

    return (
        <div style={{ height: '100%', padding: '24px', backgroundColor: '#020617', overflow: 'auto' }}>
            <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '24px' }}>System Analytics</h2>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                {/* N-grams Learned */}
                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <h3 style={{ fontSize: '14px', color: '#94a3b8', fontWeight: 500 }}>N-grams Learned</h3>
                        <DatabaseIcon size={20} style={{ color: '#3b82f6' }} />
                    </div>
                    <div style={{ fontSize: '32px', fontWeight: 700, color: '#e2e8f0' }}>
                        {status?.ngrams_learned?.toLocaleString() || '0'}
                    </div>
                </div>

                {/* Memory Usage */}
                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <h3 style={{ fontSize: '14px', color: '#94a3b8', fontWeight: 500 }}>Memory Usage</h3>
                        <Cpu size={20} style={{ color: '#8b5cf6' }} />
                    </div>
                    <div style={{ fontSize: '32px', fontWeight: 700, color: '#e2e8f0' }}>
                        {status?.memory_usage_mb?.toFixed(1) || '0'} <span style={{ fontSize: '18px', color: '#94a3b8' }}>MB</span>
                    </div>
                </div>

                {/* Uptime */}
                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <h3 style={{ fontSize: '14px', color: '#94a3b8', fontWeight: 500 }}>Uptime</h3>
                        <Zap size={20} style={{ color: '#10b981' }} />
                    </div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: '#e2e8f0' }}>
                        {status?.uptime_seconds ? formatUptime(status.uptime_seconds) : '0h 0m 0s'}
                    </div>
                </div>

                {/* Training Status */}
                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <h3 style={{ fontSize: '14px', color: '#94a3b8', fontWeight: 500 }}>Training Status</h3>
                        <TrendingUp size={20} style={{ color: status?.is_training ? '#22c55e' : '#64748b' }} />
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: 700, color: status?.is_training ? '#22c55e' : '#64748b' }}>
                        {status?.is_training ? 'Active' : 'Idle'}
                    </div>
                </div>
            </div>

            {/* Additional Info */}
            <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '24px', border: '1px solid #1e293b' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>System Information</h3>
                <div style={{ display: 'grid', gap: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', backgroundColor: '#1e293b', borderRadius: '8px' }}>
                        <span style={{ fontSize: '14px', color: '#94a3b8' }}>API Endpoint</span>
                        <span style={{ fontSize: '14px', color: '#e2e8f0', fontFamily: 'monospace' }}>
                            {import.meta.env.VITE_API_URL || 'http://localhost:8000'}
                        </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', backgroundColor: '#1e293b', borderRadius: '8px' }}>
                        <span style={{ fontSize: '14px', color: '#94a3b8' }}>Model Type</span>
                        <span style={{ fontSize: '14px', color: '#e2e8f0' }}>VoxelLM Neural/Voxel Hybrid</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', backgroundColor: '#1e293b', borderRadius: '8px' }}>
                        <span style={{ fontSize: '14px', color: '#94a3b8' }}>N-grams Learned</span>
                        <span style={{ fontSize: '14px', color: '#e2e8f0' }}>
                            {status?.ngrams_learned?.toLocaleString() || 'N/A'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
