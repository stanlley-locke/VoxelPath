import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { apiClient } from '../lib/api-client';
import type { ChatMessage, ModelInfo } from '../types/api';

export default function ChatInterface() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Model selection
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [loadedModel, setLoadedModel] = useState<string>('');
    const [modelLoading, setModelLoading] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const response = await apiClient.listModels();
            setModels(response.models || []);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    };

    const handleLoadModel = async () => {
        if (!selectedModel) return;

        setModelLoading(true);
        try {
            await apiClient.loadModel(selectedModel);
            setLoadedModel(selectedModel);
            setMessages([]);
        } catch (error) {
            alert('Failed to load model: ' + (error as Error).message);
        } finally {
            setModelLoading(false);
        }
    };

    const handleSend = async () => {
        if (!input.trim() || isLoading || !loadedModel) return;

        const userMessage: ChatMessage = {
            role: 'user',
            content: input.trim(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await apiClient.chat({
                messages: [...messages, userMessage],
                temperature: 0.85,
                max_length: 200,
            });

            setMessages(prev => [...prev, response.message]);
        } catch (error) {
            console.error('Chat error:', error);
            setMessages(prev => [
                ...prev,
                {
                    role: 'assistant',
                    content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', backgroundColor: '#020617' }}>
            {/* Model Selection Bar */}
            <div style={{ padding: '16px 24px', backgroundColor: '#0f172a', borderBottom: '1px solid #1e293b', flexShrink: 0 }}>
                <div style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', gap: '12px', alignItems: 'center' }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>
                            Select Model
                        </label>
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            disabled={modelLoading}
                            style={{
                                width: '100%',
                                padding: '10px 14px',
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '8px',
                                fontSize: '14px',
                                color: '#f1f5f9',
                                cursor: 'pointer'
                            }}
                        >
                            <option value="">-- Select a model --</option>
                            {models.map((model) => (
                                <option key={model.name} value={model.name}>
                                    {model.name} ({model.size_mb?.toFixed(1) || '?'} MB)
                                </option>
                            ))}
                        </select>
                    </div>

                    <div style={{ paddingTop: '20px' }}>
                        <button
                            onClick={handleLoadModel}
                            disabled={!selectedModel || modelLoading || selectedModel === loadedModel}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: selectedModel && selectedModel !== loadedModel ? '#2563eb' : '#334155',
                                color: '#ffffff',
                                border: 'none',
                                borderRadius: '8px',
                                fontSize: '14px',
                                fontWeight: 500,
                                cursor: selectedModel && selectedModel !== loadedModel ? 'pointer' : 'not-allowed',
                                opacity: selectedModel && selectedModel !== loadedModel ? 1 : 0.6
                            }}
                        >
                            {modelLoading ? 'Loading...' : 'Load Model'}
                        </button>
                    </div>

                    {loadedModel && (
                        <div style={{ paddingTop: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <CheckCircle size={18} style={{ color: '#22c55e' }} />
                            <span style={{ fontSize: '13px', color: '#22c55e', fontWeight: 500 }}>
                                {loadedModel}
                            </span>
                        </div>
                    )}
                </div>
            </div>

            {/* Messages */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '32px' }}>
                {!loadedModel ? (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <div style={{ textAlign: 'center' }}>
                            <AlertCircle size={48} style={{ margin: '0 auto 16px', color: '#64748b' }} />
                            <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#94a3b8', marginBottom: '8px' }}>
                                No Model Loaded
                            </h3>
                            <p style={{ fontSize: '14px', color: '#64748b' }}>
                                Please select and load a model from the dropdown above to start chatting.
                            </p>
                        </div>
                    </div>
                ) : messages.length === 0 ? (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <div style={{ textAlign: 'center', color: '#64748b' }}>
                            <p style={{ fontSize: '15px' }}>Start a conversation with {loadedModel}</p>
                        </div>
                    </div>
                ) : (
                    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
                        {messages.map((message, index) => (
                            <div key={index} style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
                                <div style={{
                                    width: '40px',
                                    height: '40px',
                                    borderRadius: '8px',
                                    backgroundColor: '#1e293b',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flexShrink: 0
                                }}>
                                    <span style={{ fontSize: '14px', fontWeight: 500 }}>
                                        {message.role === 'user' ? 'U' : 'A'}
                                    </span>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '6px' }}>
                                        {message.role === 'user' ? 'You' : 'Assistant'}
                                    </div>
                                    <div style={{ fontSize: '15px', color: '#e2e8f0', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                                        {message.content}
                                    </div>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div style={{ display: 'flex', gap: '16px' }}>
                                <div style={{
                                    width: '40px',
                                    height: '40px',
                                    borderRadius: '8px',
                                    backgroundColor: '#1e293b',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }}>
                                    <span style={{ fontSize: '14px', fontWeight: 500 }}>A</span>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: '13px', color: '#64748b', marginBottom: '6px' }}>Assistant</div>
                                    <Loader2 style={{ width: '20px', height: '20px', color: '#94a3b8' }} className="animate-spin" />
                                </div>
                            </div>
                        )}
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div style={{ borderTop: '1px solid #1e293b', padding: '20px', backgroundColor: '#0f172a' }}>
                <div style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', gap: '12px' }}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={loadedModel ? "Type a message..." : "Load a model first..."}
                        disabled={!loadedModel}
                        style={{
                            flex: 1,
                            padding: '14px 18px',
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '10px',
                            fontSize: '15px',
                            color: '#ffffff',
                            outline: 'none'
                        }}
                        onFocus={(e) => loadedModel && (e.target.style.borderColor = '#2563eb')}
                        onBlur={(e) => e.target.style.borderColor = '#334155'}
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading || !input.trim() || !loadedModel}
                        style={{
                            padding: '14px 24px',
                            backgroundColor: '#2563eb',
                            color: '#ffffff',
                            borderRadius: '10px',
                            fontSize: '15px',
                            fontWeight: 500,
                            border: 'none',
                            cursor: (isLoading || !input.trim() || !loadedModel) ? 'not-allowed' : 'pointer',
                            opacity: (isLoading || !input.trim() || !loadedModel) ? 0.5 : 1,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            transition: 'background-color 0.15s'
                        }}
                        onMouseEnter={(e) => !isLoading && input.trim() && loadedModel && (e.currentTarget.style.backgroundColor = '#1d4ed8')}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#2563eb')}
                    >
                        {isLoading ? (
                            <Loader2 style={{ width: '18px', height: '18px' }} className="animate-spin" />
                        ) : (
                            <Send style={{ width: '18px', height: '18px' }} />
                        )}
                        Send
                    </button>
                </div>
            </div>
        </div>
    );
}
