import { useState, useEffect } from 'react';
import { Download, Upload, Trash2, Save, RefreshCw } from 'lucide-react';
import { apiClient } from '../lib/api-client';
import type { ModelInfo } from '../types/api';

export default function ModelsInterface() {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [saveName, setSaveName] = useState('');

    const loadModels = async () => {
        setLoading(true);
        try {
            const response = await apiClient.listModels();
            setModels(response.models || []);
        } catch (error) {
            console.error('Failed to load models:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadModels();
    }, []);

    const handleSave = async () => {
        if (!saveName.trim()) return;

        try {
            await apiClient.saveModel(saveName.trim());
            alert('Model saved successfully!');
            setSaveName('');
            loadModels();
        } catch (error) {
            alert('Failed to save model: ' + (error as Error).message);
        }
    };

    const handleLoad = async (name: string) => {
        try {
            await apiClient.loadModel(name);
            alert('Model loaded successfully!');
        } catch (error) {
            alert('Failed to load model: ' + (error as Error).message);
        }
    };

    const handleExport = (name: string) => {
        window.open(`${import.meta.env.VITE_API_URL}/models/export/${name}`, '_blank');
    };

    return (
        <div style={{ height: '100%', display: 'flex', gap: '24px', padding: '24px', backgroundColor: '#020617', overflow: 'auto' }}>
            {/* Left Panel - Save New Model */}
            <div style={{ width: '400px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                        <Save size={18} style={{ color: '#94a3b8' }} />
                        <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Save Current Model</h3>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                        <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#94a3b8' }}>
                            Model Name
                        </label>
                        <input
                            type="text"
                            value={saveName}
                            onChange={(e) => setSaveName(e.target.value)}
                            placeholder="my-model"
                            style={{
                                width: '100%',
                                padding: '12px',
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '8px',
                                fontSize: '14px',
                                color: '#f1f5f9'
                            }}
                        />
                    </div>

                    <button
                        onClick={handleSave}
                        disabled={!saveName.trim()}
                        style={{
                            width: '100%',
                            padding: '12px',
                            backgroundColor: '#2563eb',
                            color: '#ffffff',
                            border: 'none',
                            borderRadius: '8px',
                            fontSize: '14px',
                            fontWeight: 500,
                            cursor: saveName.trim() ? 'pointer' : 'not-allowed',
                            opacity: saveName.trim() ? 1 : 0.5,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px'
                        }}
                    >
                        <Save size={16} />
                        Save Model
                    </button>
                </div>

                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#94a3b8' }}>
                        Model Info
                    </h4>
                    <div style={{ fontSize: '13px', color: '#64748b', lineHeight: '1.6' }}>
                        <p>Save your trained model to load it later or export it for deployment.</p>
                        <p style={{ marginTop: '8px' }}>Models include all n-gram weights and vocabulary.</p>
                    </div>
                </div>
            </div>

            {/* Right Panel - Model List */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ fontSize: '18px', fontWeight: 600 }}>Saved Models</h2>
                    <button
                        onClick={loadModels}
                        style={{
                            padding: '10px 16px',
                            backgroundColor: '#1e293b',
                            color: '#94a3b8',
                            border: 'none',
                            borderRadius: '8px',
                            fontSize: '14px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px'
                        }}
                    >
                        <RefreshCw size={16} />
                        Refresh
                    </button>
                </div>

                {loading ? (
                    <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                        Loading models...
                    </div>
                ) : models.length === 0 ? (
                    <div style={{
                        backgroundColor: '#0f172a',
                        borderRadius: '12px',
                        padding: '40px',
                        border: '1px solid #1e293b',
                        textAlign: 'center',
                        color: '#64748b'
                    }}>
                        No saved models yet
                    </div>
                ) : (
                    <div style={{ display: 'grid', gap: '12px' }}>
                        {models.map((model) => (
                            <div
                                key={model.name}
                                style={{
                                    backgroundColor: '#0f172a',
                                    borderRadius: '12px',
                                    padding: '16px',
                                    border: '1px solid #1e293b',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}
                            >
                                <div style={{ flex: 1 }}>
                                    <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '6px' }}>
                                        {model.name}
                                    </h3>
                                    <div style={{ fontSize: '13px', color: '#64748b' }}>
                                        {model.size_mb ? `${model.size_mb.toFixed(2)} MB` : 'Unknown size'}
                                        {model.created && ` • ${new Date(model.created).toLocaleDateString()}`}
                                    </div>
                                </div>

                                <div style={{ display: 'flex', gap: '8px' }}>
                                    <button
                                        onClick={() => handleLoad(model.name)}
                                        style={{
                                            padding: '8px 14px',
                                            backgroundColor: '#2563eb',
                                            color: '#ffffff',
                                            border: 'none',
                                            borderRadius: '6px',
                                            fontSize: '13px',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '6px'
                                        }}
                                    >
                                        <Upload size={14} />
                                        Load
                                    </button>
                                    <button
                                        onClick={() => handleExport(model.name)}
                                        style={{
                                            padding: '8px 14px',
                                            backgroundColor: '#059669',
                                            color: '#ffffff',
                                            border: 'none',
                                            borderRadius: '6px',
                                            fontSize: '13px',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '6px'
                                        }}
                                    >
                                        <Download size={14} />
                                        Export
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
