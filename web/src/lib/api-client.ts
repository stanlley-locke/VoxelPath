import { API_CONFIG, API_ENDPOINTS } from '../config/api';
import type {
    ChatRequest,
    ChatResponse,
    ModelInfo,
    SystemStatus,
    TrainingRequest,
    TrainingStatus,
} from '../types/api';

type ModelListResponse = { models: ModelInfo[] };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_CONFIG.baseURL}${path}`, {
        ...init,
        headers: { 'Content-Type': 'application/json', ...init?.headers },
    });
    if (!response.ok) throw new Error(`API request failed (${response.status})`);
    return response.json() as Promise<T>;
}

export const apiClient = {
    chat: (body: ChatRequest) => request<ChatResponse>(API_ENDPOINTS.CHAT, { method: 'POST', body: JSON.stringify(body) }),
    listModels: () => request<ModelListResponse>(API_ENDPOINTS.MODELS_LIST),
    loadModel: (name: string) => request<unknown>(API_ENDPOINTS.MODELS_LOAD, { method: 'POST', body: JSON.stringify({ name }) }),
    saveModel: (name: string) => request<unknown>(API_ENDPOINTS.MODELS_SAVE, { method: 'POST', body: JSON.stringify({ name }) }),
    startTraining: (body: TrainingRequest) => request<unknown>(API_ENDPOINTS.TRAINING_START, { method: 'POST', body: JSON.stringify(body) }),
    getTrainingStatus: () => request<TrainingStatus>(API_ENDPOINTS.TRAINING_STATUS),
    cancelTraining: () => request<unknown>(API_ENDPOINTS.TRAINING_CANCEL, { method: 'POST' }),
    getStatus: () => request<SystemStatus>(API_ENDPOINTS.STATUS),
};
