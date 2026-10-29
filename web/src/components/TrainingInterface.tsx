import { useState, useEffect } from 'react';
import { Play, Square, Settings as SettingsIcon, Activity, Upload as UploadIcon } from 'lucide-react';
import { apiClient } from '../lib/api-client';
import type { MetricValue, TrainingConfig } from '../types/api';

export default function TrainingInterface() {
    const [corpus, setCorpus] = useState('');
    const [validationCorpus, setValidationCorpus] = useState('');
    const [isTraining, setIsTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentBatch, setCurrentBatch] = useState(0);
    const [totalBatches, setTotalBatches] = useState(0);
    const [metrics, setMetrics] = useState<Record<string, MetricValue>>({});

    // Training config
    const [batchSize, setBatchSize] = useState(10000);
    const [validationSplit, setValidationSplit] = useState(0.1);
    const [maxNgramLevel, setMaxNgramLevel] = useState(5);

    useEffect(() => {
        const interval = setInterval(async () => {
            if (isTraining) {
                try {
                    const status = await apiClient.getTrainingStatus();
                    setIsTraining(status.is_training);
                    setProgress(status.progress_percent);
                    setCurrentBatch(status.current_batch);
                    setTotalBatches(status.total_batches);
                    setMetrics(status.metrics);
                } catch (error) {
                    console.error('Failed to get status:', error);
                }
            }
        }, 1000);

        return () => clearInterval(interval);
    }, [isTraining]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, isValidation: boolean) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const text = event.target?.result as string;
            if (isValidation) {
                setValidationCorpus(text);
            } else {
                setCorpus(text);
            }
        };
        reader.readAsText(file);
    };

    const handleStartTraining = async () => {
        if (!corpus.trim()) return;

        try {
            const config: TrainingConfig = {
                batch_size: batchSize,
                validation_split: validationCorpus ? 0 : validationSplit,
                max_ngram_level: maxNgramLevel,
                detailed_logging: true,
            };

            await apiClient.startTraining({
                corpus,
                validation_corpus: validationCorpus || undefined,
                config
            });
            setIsTraining(true);
        } catch (error) {
            console.error('Training error:', error);
            alert('Failed to start training: ' + (error as Error).message);
        }
    };

    const handleCancelTraining = async () => {
        try {
            await apiClient.cancelTraining();
            setIsTraining(false);
            setProgress(0);
        } catch (error) {
            console.error('Cancel error:', error);
        }
    };

    return (
        <div style={{ height: '100%', display: 'flex', gap: '24px', padding: '24px', backgroundColor: '#020617', overflow: 'auto' }}>
            {/* Left Panel - Training Controls */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* Training Corpus */}
                <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <label style={{ fontSize: '14px', fontWeight: 500, color: '#e2e8f0' }}>
                            Training Corpus *
                        </label>
                        <label style={{
                            padding: '8px 14px',
                            backgroundColor: '#2563eb',
                            color: '#ffffff',
                            borderRadius: '6px',
                            fontSize: '13px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px'
                        }}>
                            <UploadIcon size={14} />
                            Load File
                            <input
                                type="file"
                                accept=".txt,.md"
                                onChange={(e) => handleFileUpload(e, false)}
                                disabled={isTraining}
                                style={{ display: 'none' }}
                            />
                        </label >
                    </div >
                    <textarea
                        value={corpus}
                        onChange={(e) => setCorpus(e.target.value)}
                        placeholder="Enter or upload training text..."
                        disabled={isTraining}
                        style={{
                            width: '100%',
                            height: '180px',
                            padding: '12px',
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '8px',
                            fontSize: '14px',
                            color: '#f1f5f9',
                            resize: 'vertical',
                            fontFamily: 'monospace'
                        }}
                    />
                    {
                        corpus && (
                            <div style={{ marginTop: '8px', fontSize: '12px', color: '#64748b' }}>
                                {corpus.length.toLocaleString()} characters
                            </div>
                        )
                    }
                </div >

                {/* Validation Corpus */}
                < div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <label style={{ fontSize: '14px', fontWeight: 500, color: '#e2e8f0' }}>
                            Validation Corpus (Optional)
                        </label>
                        <label style={{
                            padding: '8px 14px',
                            backgroundColor: '#059669',
                            color: '#ffffff',
                            borderRadius: '6px',
                            fontSize: '13px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px'
                        }}>
                            <UploadIcon size={14} />
                            Load File
                            <input
                                type="file"
                                accept=".txt,.md"
                                onChange={(e) => handleFileUpload(e, true)}
                                disabled={isTraining}
                                style={{ display: 'none' }}
                            />
                        </label>
                    </div>
                    <textarea
                        value={validationCorpus}
                        onChange={(e) => setValidationCorpus(e.target.value)}
                        placeholder="Optional validation text..."
                        disabled={isTraining}
                        style={{
                            width: '100%',
                            height: '120px',
                            padding: '12px',
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '8px',
                            fontSize: '14px',
                            color: '#f1f5f9',
                            resize: 'vertical',
                            fontFamily: 'monospace'
                        }}
                    />
                    {
                        validationCorpus && (
                            <div style={{ marginTop: '8px', fontSize: '12px', color: '#64748b' }}>
                                {validationCorpus.length.toLocaleString()} characters
                            </div>
                        )
                    }
                </div >

                {/* Config */}
                < div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                        <SettingsIcon size={18} style={{ color: '#94a3b8' }} />
                        <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Configuration</h3>
                    </div>

                    <div style={{ display: 'grid', gap: '12px' }}>
                        <div>
                            <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#94a3b8' }}>
                                Batch Size: {batchSize.toLocaleString()}
                            </label>
                            <input
                                type="range"
                                min="1000"
                                max="50000"
                                step="1000"
                                value={batchSize}
                                onChange={(e) => setBatchSize(Number(e.target.value))}
                                disabled={isTraining}
                                style={{ width: '100%' }}
                            />
                        </div>

                        {!validationCorpus && (
                            <div>
                                <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#94a3b8' }}>
                                    Validation Split: {(validationSplit * 100).toFixed(0)}%
                                </label>
                                <input
                                    type="range"
                                    min="0"
                                    max="0.3"
                                    step="0.05"
                                    value={validationSplit}
                                    onChange={(e) => setValidationSplit(Number(e.target.value))}
                                    disabled={isTraining}
                                    style={{ width: '100%' }}
                                />
                            </div>
                        )}

                        <div>
                            <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#94a3b8' }}>
                                Max N-gram Level: {maxNgramLevel}
                            </label>
                            <input
                                type="range"
                                min="2"
                                max="10"
                                step="1"
                                value={maxNgramLevel}
                                onChange={(e) => setMaxNgramLevel(Number(e.target.value))}
                                disabled={isTraining}
                                style={{ width: '100%' }}
                            />
                        </div>
                    </div>
                </div >

                {/* Actions */}
                < div style={{ display: 'flex', gap: '12px' }}>
                    {!isTraining ? (
                        <button
                            onClick={handleStartTraining}
                            disabled={!corpus.trim()}
                            style={{
                                flex: 1,
                                padding: '14px',
                                backgroundColor: '#2563eb',
                                color: '#ffffff',
                                border: 'none',
                                borderRadius: '10px',
                                fontSize: '15px',
                                fontWeight: 500,
                                cursor: corpus.trim() ? 'pointer' : 'not-allowed',
                                opacity: corpus.trim() ? 1 : 0.5,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px'
                            }}
                        >
                            <Play size={18} />
                            Start Training
                        </button>
                    ) : (
                        <button
                            onClick={handleCancelTraining}
                            style={{
                                flex: 1,
                                padding: '14px',
                                backgroundColor: '#dc2626',
                                color: '#ffffff',
                                border: 'none',
                                borderRadius: '10px',
                                fontSize: '15px',
                                fontWeight: 500,
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px'
                            }}
                        >
                            <Square size={18} />
                            Cancel Training
                        </button>
                    )}
                </div >
            </div >

            {/* Right Panel - Status */}
            < div style={{ width: '400px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* Progress */}
                < div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                        <Activity size={18} style={{ color: '#94a3b8' }} />
                        <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Training Progress</h3>
                    </div>

                    {
                        isTraining ? (
                            <div>
                                <div style={{ marginBottom: '12px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                        <span style={{ fontSize: '13px', color: '#94a3b8' }}>Progress</span>
                                        <span style={{ fontSize: '13px', fontWeight: 600, color: '#3b82f6' }}>
                                            {progress.toFixed(1)}%
                                        </span>
                                    </div>
                                    <div style={{ width: '100%', height: '8px', backgroundColor: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{
                                            width: `${progress}%`,
                                            height: '100%',
                                            backgroundColor: '#3b82f6',
                                            transition: 'width 0.3s'
                                        }} />
                                    </div>
                                </div>

                                <div style={{ fontSize: '13px', color: '#94a3b8' }}>
                                    Batch {currentBatch} / {totalBatches}
                                </div>
                            </div>
                        ) : (
                            <div style={{ textAlign: 'center', padding: '20px', color: '#64748b', fontSize: '14px' }}>
                                No active training
                            </div>
                        )
                    }
                </div >

                {/* Metrics */}
                {
                    Object.keys(metrics).length > 0 && (
                        <div style={{ backgroundColor: '#0f172a', borderRadius: '12px', padding: '20px', border: '1px solid #1e293b' }}>
                            <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '12px' }}>Metrics</h3>
                            <div style={{ display: 'grid', gap: '8px' }}>
                                {Object.entries(metrics).map(([key, value]) => (
                                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                                        <span style={{ color: '#94a3b8' }}>{key}:</span>
                                        <span style={{ color: '#e2e8f0', fontWeight: 500 }}>
                                            {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )
                }
            </div >
        </div >
    );
}
