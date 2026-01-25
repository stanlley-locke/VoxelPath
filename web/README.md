# VoxelLM Web Interface

Modern TypeScript web interface for VoxelLM with real-time training visualization and model management.

## Features

- 💬 **Chat Interface** - Real-time conversations with loaded models
- 🏋️ **Training Panel** - Upload corpus files, configure parameters, monitor progress
- 💾 **Model Management** - Save, load, export trained models
- 📊 **Analytics Dashboard** - Live system metrics and performance stats

## Tech Stack

- **Frontend**: React 18 + TypeScript 5
- **Build Tool**: Vite 5
- **Styling**: Inline styles (production-grade sizing)
- **API Client**: Typed fetch client
- **Icons**: Lucide React

## Quick Start

### Prerequisites

- Node.js 16+ 
- npm or yarn
- VoxelLM API server running at `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The interface will be available at `http://localhost:5173`

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Configuration

Create `.env` file in the` `web/` directory:

```env
VITE_API_URL=http://localhost:8000
```

## Project Structure

```
web/
├── src/
│   ├── components/          # React components
│   │   ├── ChatInterface.tsx       # Chat UI
│   │   ├── TrainingInterface.tsx   # Training controls
│   │   ├── ModelsInterface.tsx     # Model management
│   │   └── AnalyticsInterface.tsx  # Metrics dashboard
│   ├── lib/
│   │   └── api-client.ts    # Typed API client
│   ├── types/
│   │   └── api.ts           # TypeScript types
│   ├── config/
│   │   └── api.ts           # API configuration
│   ├── App.tsx              # Main layout
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/
├── .env
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Interface Pages

### 1. Chat

**Features:**
- Model selection dropdown
- Load model before chatting
- Message history with avatars
- Real-time responses
- Keyboard shortcuts (Enter to send)

**Usage:**
1. Select a saved model from dropdown
2. Click "Load Model"
3. Start chatting!

### 2. Training

**Features:**
- File upload for training corpus (.txt, .md)
- File upload for validation corpus (optional)
- Manual text input
- Configuration sliders:
  - Batch size (1K-50K)
  - Validation split (0-30%)
  - Max n-gram level (2-10)
- Real-time progress bar
- Live metrics display
- Training log viewer

**Usage:**
1. Upload or paste training text
2. Optionally upload validation text
3. Adjust configuration parameters
4. Click "Start Training"
5. Monitor progress in real-time

### 3. Models

**Features:**
- Save current model with custom name
- List all saved models with metadata
- Load saved models
- Export models for deployment
- Model size and creation date display

**Usage:**
1. After training, enter a model name
2. Click "Save Model"
3. View in models list
4. Load or export as needed

### 4. Analytics

**Features:**
- N-grams learned counter
- Memory usage monitor
- System uptime display
- Training status indicator
- Auto-refresh (2s interval)

**Metrics:**
- Real-time system statistics
- Live updates during training
- Performance monitoring

## API Integration

The web interface communicates with the VoxelLM API via typed client:

```typescript
import { apiClient } from './lib/api-client';

// Chat
const response = await apiClient.chat({
  messages: [{role: 'user', content: 'Hello'}]
});

// Train
await apiClient.startTraining({
  corpus: 'Training text...',
  config: { batch_size: 10000 }
});

// Status
const status = await apiClient.getTrainingStatus();
```

## Type Safety

All API responses are fully typed:

```typescript
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ModelInfo {
  name: string;
  size_mb?: number;
  created?: string;
}

interface SystemStatus {
  ngrams_learned: number;
  memory_usage_mb: number;
  uptime_seconds: number;
  is_training: boolean;
}
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style

- TypeScript strict mode enabled
- Inline styles for reliability
- Functional components with hooks
- Proper error handling
- Loading states for async operations

## Troubleshooting

### API Connection Issues

If you see "Failed to fetch" errors:

1. Ensure API server is running: `python -m voxellm.server_main`
2. Check `.env` has correct `VITE_API_URL`
3. Verify CORS is enabled on API server
4. Check browser console for errors

### Build Errors

If Vite fails to build:

1. Clear node_modules: `rm -rf node_modules`
2. Clean install: `npm install`
3. Clear Vite cache: `rm -rf node_modules/.vite`

### Styling Issues

All components use inline styles for production reliability. If elements appear tiny:

1. Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. Clear browser cache
3. Check `index.html` has proper viewport meta tag

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Contributing

When adding new features:

1. Follow existing component patterns
2. Use TypeScript for type safety
3. Add proper error handling
4. Update API client types
5. Test with real API server

## License

Same as parent project (MIT)

---

**Built with React + TypeScript + Vi te**
