import { useEffect, useState } from 'react'
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  CircleDot,
  Copy,
  Github,
  Hexagon,
  Menu,
  Network,
  Orbit,
  PanelTop,
  Search,
  Sparkles,
  X,
  Zap,
} from 'lucide-react'
import './App.css'

const sections = [
  { id: 'overview', label: 'Overview' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'inference', label: 'Inference' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'performance', label: 'Performance' },
]

const architecture = [
  { title: 'Voxel engine', text: '3D coordinate mapping, density learning, and path-based inference.', icon: Hexagon, color: 'violet' },
  { title: 'Neural scorer', text: 'Momentum-weighted n-gram predictions for a hybrid generation signal.', icon: Sparkles, color: 'cyan' },
  { title: 'RAG system', text: 'Document chunking, embedding retrieval, and context-aware generation.', icon: Search, color: 'amber' },
  { title: 'FastAPI server', text: 'RESTful endpoints, live training status, and model management.', icon: Network, color: 'green' },
]

const codeSnippet = `def get_coords(trigram):
    """Maps a 3-char string to (x, y, z) ASCII coordinates."""
    return tuple(ord(c) if ord(c) < 128 else 0 for c in trigram)

print(get_coords("The"))
# Output: (84, 104, 101)`

function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const [activeSection, setActiveSection] = useState('overview')

  useEffect(() => {
    const handleScroll = () => {
      const current = sections
        .slice()
        .reverse()
        .find(({ id }) => window.scrollY >= (document.getElementById(id)?.offsetTop ?? 0) - 160)
      if (current) setActiveSection(current.id)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const copySnippet = async () => {
    await navigator.clipboard.writeText(codeSnippet)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1800)
  }

  const navigate = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setMenuOpen(false)
  }

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#overview" onClick={() => navigate('overview')}>
          <span className="brand-mark"><Hexagon size={20} strokeWidth={2.2} /></span>
          <span>Voxel<span>Path</span></span>
        </a>
        <nav className={`main-nav ${menuOpen ? 'is-open' : ''}`}>
          {sections.map((section) => (
            <button className={activeSection === section.id ? 'active' : ''} key={section.id} onClick={() => navigate(section.id)}>
              {section.label}
            </button>
          ))}
          <a className="nav-cta" href="https://github.com/stanlley-locke/VoxelPath" target="_blank" rel="noreferrer">
            View on GitHub <ArrowUpRight size={15} />
          </a>
        </nav>
        <button className="menu-toggle" aria-label="Toggle navigation" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X size={21} /> : <Menu size={21} />}
        </button>
      </header>

      <main>
        <section className="hero section-wrap" id="overview">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-dot" /> Research whitepaper · January 2026</div>
            <h1>Language as a <em>trajectory</em>, not a token.</h1>
            <p className="hero-lede">VoxelPath is a geometric 3D coordinate-mapped model for efficient pattern recognition and predictive modeling.</p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={() => navigate('methodology')}>Explore the model <ChevronRight size={17} /></button>
              <a className="button button-quiet" href="#implementation">Read implementation <ArrowUpRight size={16} /></a>
            </div>
            <div className="author-line"><span>By Stanley Wasonga</span><span className="line" /><span>BSc Cloud Computing</span></div>
          </div>
          <div className="hero-visual" aria-label="Illustration of the VoxelPath coordinate space">
            <div className="visual-glow" />
            <div className="cube cube-back" />
            <div className="cube cube-front" />
            <div className="axis axis-x"><span>x</span></div>
            <div className="axis axis-y"><span>y</span></div>
            <div className="axis axis-z"><span>z</span></div>
            <div className="point point-one"><span>84, 104, 101</span></div>
            <div className="point point-two" />
            <div className="point point-three" />
            <div className="trace trace-one" />
            <div className="trace trace-two" />
            <div className="visual-caption"><CircleDot size={13} /> Every trigram becomes a point in space</div>
          </div>
        </section>

        <section className="stat-strip section-wrap">
          <div><strong>128³</strong><span>coordinate space</span></div>
          <div><strong>O(n)</strong><span>training complexity</span></div>
          <div><strong>8.4 MB</strong><span>dense grid footprint</span></div>
          <div><strong>0–127</strong><span>ASCII coordinate range</span></div>
        </section>

        <section className="content-section section-wrap" id="methodology">
          <div className="section-kicker">01 / Methodology</div>
          <div className="section-heading">
            <h2>A cube that remembers<br /><span>how language moves.</span></h2>
            <p>VoxelPath replaces abstract semantic vectors with a transparent geometric encoding. The intrinsic numerical properties of the 128-character ASCII set provide a fixed coordinate base for every three-character sequence.</p>
          </div>
          <div className="method-grid">
            <div className="feature-card coordinate-card">
              <div className="card-icon violet"><Orbit size={21} /></div>
              <h3>The voxel cube</h3>
              <p>A 3D tensor V ∈ ℝ<sup>128×128×128</sup> is the universe for all possible ASCII trigrams.</p>
              <div className="coordinate-display"><span>Ab1</span><b>→</b><span className="coord">(65, 98, 49)</span></div>
              <p className="card-note">The first, second, and third characters map directly to the x, y, and z axes.</p>
            </div>
            <div className="feature-card density-card">
              <div className="density-bars"><i /><i /><i /><i /><i /><i /><i /></div>
              <div className="card-icon cyan"><Zap size={21} /></div>
              <h3>Density is memory</h3>
              <p>Every voxel stores the frequency of its trigram. Common patterns become dense, high-heat clusters; rare noise remains in the cold, empty space.</p>
              <div className="heat-legend"><span><i className="hot" /> frequent</span><span><i className="cold" /> rare</span></div>
            </div>
          </div>
          <div className="code-panel">
            <div className="code-top"><span><i /> Coordinate mapping</span><button onClick={copySnippet}>{copied ? <Check size={14} /> : <Copy size={14} />} {copied ? 'Copied' : 'Copy'}</button></div>
            <pre><code>{codeSnippet}</code></pre>
          </div>
        </section>

        <section className="dark-section">
          <div className="section-wrap content-section" id="inference">
            <div className="section-kicker">02 / Learning & inference</div>
            <div className="section-heading light">
              <h2>Training draws the map.<br /><span>Inference finds the route.</span></h2>
              <p>As a one-character window moves across a corpus, each point connects to the next and forms a continuous wire through the cube. Prediction becomes navigation across the paths that have proved most useful.</p>
            </div>
            <div className="steps-grid">
              <article className="step-card"><span className="step-number">01</span><h3>Trace formation</h3><p>For a string of length L, the system generates L−2 spatial points. “Hello” creates three linked coordinates: <b>Hel</b>, <b>ell</b>, and <b>llo</b>.</p><div className="mini-path"><span>Hel</span><i /><span>ell</span><i /><span>llo</span></div></article>
              <article className="step-card"><span className="step-number">02</span><h3>Density-weighted search</h3><p>A* traversal treats high-frequency voxels as low-resistance routes. The cost function is <b>g(n) = 1 / (density[n] + 1)</b>.</p><div className="formula">common path <span>→</span> less resistance</div></article>
              <article className="step-card"><span className="step-number">03</span><h3>Geometric momentum</h3><p>The vector between the last two points establishes velocity. When the trajectory is linear, the engine biases the next prediction along that direction.</p><div className="momentum-line"><span>•</span><i /><i /><i /><b>→</b></div></article>
            </div>
          </div>
        </section>

        <section className="content-section section-wrap" id="architecture">
          <div className="section-kicker">03 / Advanced architecture</div>
          <div className="section-heading">
            <h2>Geometry at the core.<br /><span>Practicality at every layer.</span></h2>
            <p>VoxelPath adds architectural features that extend a deterministic geometric engine into a useful, production-minded language system.</p>
          </div>
          <div className="architecture-grid">
            {architecture.map(({ title, text, icon: Icon, color }, index) => (
              <article className="architecture-card" key={title}>
                <div className={`card-icon ${color}`}><Icon size={20} /></div><span className="card-index">0{index + 1}</span><h3>{title}</h3><p>{text}</p>
              </article>
            ))}
          </div>
          <div className="feature-list">
            <div className="feature-row"><span className="feature-symbol">◌</span><div><h3>Semantic halo</h3><p>Parallel layers at an offset of 32 link uppercase and lowercase characters, introducing case-insensitive proximity without abandoning the coordinate model.</p></div><span className="feature-tag">proximity</span></div>
            <div className="feature-row"><span className="feature-symbol">↗</span><div><h3>Trajectory tunneling</h3><p>Frequent paths that exceed a threshold become direct vector links, allowing the engine to jump across familiar words and compress common knowledge.</p></div><span className="feature-tag">macro-jumps</span></div>
            <div className="feature-row"><span className="feature-symbol">⌁</span><div><h3>Temporal decay</h3><p>A periodic decay function evaporates one-off noise and typos while retaining established structural knowledge on limited hardware.</p></div><span className="feature-tag">self-cleaning</span></div>
          </div>
        </section>

        <section className="performance-section" id="performance">
          <div className="section-wrap content-section">
            <div className="section-kicker">04 / Efficiency</div>
            <div className="performance-layout">
              <div className="section-heading light"><h2>Small footprint.<br /><span>Visible decisions.</span></h2><p>VoxelPath is designed for edge devices, IoT sensors, and standard consumer hardware where memory, latency, and explainability matter.</p><a className="text-link" href="#implementation" onClick={() => navigate('implementation')}>See the full implementation <ArrowUpRight size={15} /></a></div>
              <div className="comparison-table"><div className="table-head"><span>Metric</span><span>Transformer / LLM</span><span>VoxelPath</span></div><div><span>Complexity</span><span>O(n²) attention</span><strong>O(n) direct indexing</strong></div><div><span>Memory</span><span>GBs to TBs</span><strong>~8.4 MB</strong></div><div><span>Training</span><span>GPU intensive</span><strong>CPU bound</strong></div><div><span>Explainability</span><span>Black box</span><strong>Visualizable path</strong></div></div>
            </div>
          </div>
        </section>

        <section className="implementation-section section-wrap" id="implementation">
          <div className="section-kicker">05 / Implementation · VoxelLM</div>
          <div className="implementation-heading"><h2>The research, made runnable.</h2><p>VoxelLM implements the VoxelPath paradigm with a hybrid neural scorer, retrieval-augmented generation, a modern API, and a live web interface.</p></div>
          <div className="implementation-grid"><div className="install-card"><div className="terminal-top"><span /><span /><span /></div><pre><code><span className="muted">$</span> git clone github.com/stanlley-locke/VoxelPath{'\n'}<span className="muted">$</span> cd VoxelPath/web{'\n'}<span className="muted">$</span> npm install{'\n'}<span className="muted">$</span> npm run dev</code></pre><a href="https://github.com/stanlley-locke/VoxelPath" target="_blank" rel="noreferrer">Open repository <ArrowUpRight size={14} /></a></div><div className="implementation-points"><div><PanelTop size={18} /><span><b>Modern web interface</b>React + TypeScript with live training visualization, chat, models, and analytics.</span></div><div><Network size={18} /><span><b>RAG integration</b>Document retrieval and context-enhanced generation for grounded output.</span></div><div><Github size={18} /><span><b>Open and inspectable</b>MIT-licensed research implementation built for experimentation.</span></div></div></div>
        </section>

        <section className="closing-section"><div className="section-wrap"><div className="closing-mark"><Hexagon size={27} /></div><h2>A different coordinate<br /><em>system for intelligence.</em></h2><p>VoxelPath represents a shift toward deterministic spatial AI: lightweight, transparent, and efficient enough to take pattern recognition beyond the data center.</p><a className="button button-primary" href="https://github.com/stanlley-locke/VoxelPath" target="_blank" rel="noreferrer">Explore VoxelLM on GitHub <ArrowUpRight size={16} /></a><div className="footer-meta"><span>VoxelPath Whitepaper</span><span>Stanley Wasonga · January 17, 2026</span><span>MIT License</span></div></div></section>
      </main>
    </div>
  )
}

export default App
