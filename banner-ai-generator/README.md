# 🎨 Multi AI Agent Banner Generator

Hệ thống tạo banner quảng cáo tự động sử dụng kiến trúc Multi AI Agent tiên tiến với khả năng phân tích chiến lược, thiết kế background/foreground, phát triển code và đánh giá chất lượng tự động.

## 🌟 Tính Năng Nổi Bật

- 🤖 **5 AI Agents chuyên biệt**: Strategist, Background Designer, Foreground Designer, Developer, Design Reviewer
- 🎯 **Tự động hóa hoàn toàn**: Từ brief đến banner hoàn chỉnh
- 🚀 **Real-time monitoring**: Theo dõi tiến độ và hiệu suất hệ thống
- 🎨 **Đa định dạng xuất**: SVG, HTML/CSS, PNG, Figma Plugin
- 📱 **Web Interface hiện đại**: Vue.js với real-time updates
- 🔄 **ReAct Pattern**: Self-refinement cho chất lượng tối ưu

## 🏗️ Kiến Trúc Hệ Thống

### Core Infrastructure
- **Memory Manager**: Shared memory và session management
- **Communication System**: Agent coordination với message queue và event dispatcher
- **Configuration Management**: Centralized configuration cho agents và models

### AI Agents Pipeline
```
Campaign Brief → [Strategist] → [Background Designer] → [Foreground Designer] → [Developer] → [Design Reviewer] → Final Banner
```

1. **🧠 Strategist Agent**: 
   - Phân tích campaign brief và strategic direction
   - Xử lý logo và brand assets
   - Logo color extraction và brand analysis

2. **🎨 Background Designer Agent**:
   - Text-to-Image generation với FLUX.1-schnell
   - ReAct pattern cho iterative refinement
   - Background optimization và quality assessment

3. **📐 Foreground Designer Agent**:
   - AI-powered layout engine
   - Typography management và font optimization
   - Component placement với accessibility compliance
   - JSON blueprint generation

4. **💻 Developer Agent**:
   - Multi-format code generation (SVG, HTML/CSS, Figma)
   - Code optimization và minification
   - Responsive design implementation

5. **🔍 Design Reviewer Agent**:
   - Multi-criteria quality evaluation
   - Brand compliance checking
   - Accessibility auditing (WCAG 2.1)
   - Performance evaluation

### AI Models Integration
- **LLM Interface**: OpenAI GPT-4, Anthropic Claude
- **T2I Interface**: FLUX.1-schnell, DALL-E 3
- **MLLM Interface**: Multimodal models cho design review

### Backend & Frontend
- **REST API**: FastAPI với async processing và WebSocket
- **Web Frontend**: Vue.js 3 với Pinia state management
- **Database**: SQLAlchemy ORM với PostgreSQL
- **File Processing**: Comprehensive asset handling và validation

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
Node.js 18+
Redis Server
PostgreSQL (optional, SQLite for development)
```

### Installation

1. **Clone repository**
```bash
git clone <repository-url>
cd banner-ai-generator
```

2. **Setup Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup Frontend**
```bash
cd frontend
npm install
```

4. **Environment Configuration**
```bash
cp config/.env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
# REDIS_URL=redis://localhost:6379
# DATABASE_URL=postgresql://user:pass@localhost/banner_ai_db
```

5. **Database Setup**
```bash
alembic upgrade head
```

6. **Start Services**
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start API Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start Frontend
cd frontend && npm run dev

# Terminal 4: Start Communication System (optional)
python -m communication.agent_coordinator
```

### Access the Application
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc

## 📊 Workflow Example

### 1. Banner Creation via Web Interface
1. Navigate to "Create Banner" page
2. Fill campaign brief với company name, product, target audience
3. Upload brand assets (logo, images)
4. Set design preferences (style, colors, layout)
5. Click "Generate Banner"
6. Monitor real-time progress
7. Download completed banner in multiple formats

### 2. Programmatic Usage
```python
from api.services.banner_service import BannerService

# Create banner request
banner_request = {
    "name": "Summer Sale Campaign",
    "campaign_brief": "Promote 50% off summer collection for fashion brand",
    "brand_guidelines": {
        "colors": "#FF6B6B, #4ECDC4, #45B7D1",
        "fonts": "Inter, Roboto",
        "voice_tone": "friendly"
    },
    "target_audience": {
        "age_group": "25-34",
        "interests": "fashion, lifestyle"
    },
    "design_preferences": {
        "style": "modern",
        "color_scheme": "vibrant",
        "layout": "centered",
        "size": "leaderboard"
    }
}

# Generate banner
banner_service = BannerService()
banner = await banner_service.create_banner(banner_request)

# Monitor progress
status = await banner_service.get_banner_status(banner.id)
print(f"Progress: {status.progress_percentage}%")
```

## 🛠️ Configuration

### Agent Configuration
```python
# config/agent_config.py
STRATEGIST_CONFIG = {
    "brief_analyzer": {
        "llm_model": "gpt-4-turbo-preview",
        "analysis_depth": "comprehensive"
    },
    "logo_processor": {
        "max_size": (512, 512),
        "color_extraction_algorithm": "kmeans",
        "output_formats": ["PNG", "SVG"]
    }
}

BACKGROUND_DESIGNER_CONFIG = {
    "t2i_model": "flux-1-schnell",
    "react_iterations": 5,
    "quality_threshold": 0.85,
    "style_transfer_enabled": True
}

FOREGROUND_DESIGNER_CONFIG = {
    "layout_engine": "ai_powered",
    "typography_optimization": True,
    "accessibility_compliance": "wcag_2.1_aa"
}
```

### Model Configuration
```python
# config/model_config.py
AI_MODELS = {
    "llm": {
        "openai_gpt4": {
            "model": "gpt-4-turbo-preview",
            "max_tokens": 4000,
            "temperature": 0.7
        },
        "anthropic_claude": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 4000
        }
    },
    "t2i": {
        "flux_schnell": {
            "model_path": "black-forest-labs/FLUX.1-schnell",
            "guidance_scale": 7.5,
            "num_inference_steps": 4
        }
    }
}
```

## 📈 API Endpoints

### Campaign Management
```http
POST   /api/campaigns/              # Create new campaign
GET    /api/campaigns/              # List campaigns
GET    /api/campaigns/{id}          # Get campaign details
PUT    /api/campaigns/{id}          # Update campaign
DELETE /api/campaigns/{id}          # Delete campaign
```

### Design Generation
```http
POST   /api/designs/generate        # Generate new banner
GET    /api/designs/               # List designs
GET    /api/designs/{id}           # Get design details
GET    /api/designs/{id}/progress  # Get generation progress
POST   /api/designs/{id}/iterate   # Request design iteration
GET    /api/designs/{id}/download/{format}  # Download design
```

### Asset Management
```http
POST   /api/assets/upload          # Upload assets
GET    /api/assets/{id}            # Get asset details
POST   /api/assets/validate        # Validate uploaded assets
POST   /api/assets/optimize        # Optimize images
```

### System Monitoring
```http
GET    /api/system/status          # System health
GET    /api/agents/               # Agent status
GET    /api/workflows/            # Active workflows
POST   /api/agents/{id}/restart   # Restart agent
```

### WebSocket Endpoints
```http
WS     /ws/designs/{id}/progress   # Real-time progress updates
WS     /ws/system/events          # System events stream
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# API tests
pytest tests/api/ -v

# End-to-end tests
pytest tests/e2e/ -v

# With coverage
pytest --cov=. --cov-report=html
```

### Example Test
```python
# tests/test_banner_generation.py
async def test_banner_generation_workflow():
    """Test complete banner generation workflow"""
    
    # Create banner request
    request = BannerCreateRequest(
        name="Test Banner",
        campaign_brief="Test campaign for unit testing"
    )
    
    # Generate banner
    banner = await banner_service.create_banner(request)
    assert banner.status == "processing"
    
    # Wait for completion
    await wait_for_completion(banner.id, timeout=300)
    
    # Verify outputs
    completed_banner = await banner_service.get_banner(banner.id)
    assert completed_banner.status == "completed"
    assert completed_banner.generated_code is not None
    assert completed_banner.preview_urls is not None
```

## 🐳 Docker Deployment

### Development
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production
```bash
# Production deployment with scaling
docker-compose -f docker-compose.prod.yml up -d

# Scale API servers
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
kubectl get pods -n banner-ai
```

## 📊 Monitoring & Analytics

### System Health
```http
GET /health                    # Overall health check
GET /health/agents            # AI agents health
GET /health/dependencies      # External dependencies
```

### Metrics Dashboard
- **Generation Statistics**: Success rates, processing times
- **Agent Performance**: Individual agent metrics
- **Resource Usage**: CPU, memory, queue sizes
- **Error Tracking**: Failed generations và error analysis

### Real-time Monitoring
```python
# WebSocket connection for real-time events
websocket_url = "ws://localhost:8000/ws/system/events"

async with websockets.connect(websocket_url) as websocket:
    async for message in websocket:
        event = json.loads(message)
        print(f"Event: {event['type']}, Data: {event['data']}")
```

## 🔧 Advanced Features

### Custom Agent Development
```python
# Extend existing agents
class CustomStrategistAgent(StrategistAgent):
    async def custom_industry_analysis(self, brief: str) -> Dict:
        """Custom industry-specific analysis"""
        # Your custom logic here
        return analysis

# Register custom agent
agent_registry.register("custom_strategist", CustomStrategistAgent)
```

### Workflow Customization
```python
from communication.agent_coordinator import WorkflowStep

# Create custom workflow
custom_workflow = [
    WorkflowStep("strategist", "strategist", "analyze_brief"),
    WorkflowStep("custom_analyzer", "custom_strategist", "industry_analysis"),
    WorkflowStep("background_designer", "background_designer", "generate_background"),
    # ... more steps
]

coordinator.register_workflow("custom_banner_generation", custom_workflow)
```

### Plugin System
```python
# Create design plugin
class CustomDesignPlugin:
    def enhance_design(self, design_data: Dict) -> Dict:
        """Enhance design with custom logic"""
        # Custom enhancement logic
        return enhanced_design

# Register plugin
plugin_manager.register("custom_enhancer", CustomDesignPlugin())
```

## 🚦 Performance Optimization

### Caching Strategy
- **Redis Cache**: Agent results và intermediate data
- **File Cache**: Generated images và assets
- **Database Cache**: Frequently accessed data

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  api:
    scale: 3
  
  background_designer:
    scale: 2
    
  foreground_designer:
    scale: 2
```

### Load Balancing
```nginx
# nginx.conf
upstream api_servers {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_servers;
    }
}
```

## 🤝 Contributing

### Development Setup
1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes và add tests
5. Run quality checks: `pre-commit run --all-files`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push branch: `git push origin feature/amazing-feature`
8. Create Pull Request

### Code Style Guidelines
- **Python**: Follow PEP 8, use Black formatter
- **TypeScript**: Follow ESLint rules, use Prettier
- **Vue**: Follow Vue.js style guide
- **Documentation**: Update docs for new features

### Testing Requirements
- Unit tests for all new functions
- Integration tests for new endpoints
- E2E tests for new user workflows
- Performance tests for critical paths

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙋‍♂️ Support & Community

- 📧 **Email**: support@banner-ai-generator.com
- 💬 **Discord**: [Join our community](https://discord.gg/banner-ai)
- 📖 **Documentation**: [docs.banner-ai-generator.com](https://docs.banner-ai-generator.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/org/banner-ai-generator/issues)
- 📝 **Wiki**: [Project Wiki](https://github.com/org/banner-ai-generator/wiki)

## 🗺️ Roadmap

### ✅ Phase 1 - Core System (Completed)
- [x] Multi-agent architecture implementation
- [x] Complete workflow từ brief đến banner
- [x] Web interface với real-time monitoring
- [x] REST API với comprehensive endpoints
- [x] Database integration và file processing

### 🚧 Phase 2 - Enhanced Features (Q1 2024)
- [ ] Advanced AI model integration
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] A/B testing capabilities
- [ ] Brand consistency scoring

### 🔮 Phase 3 - Enterprise Features (Q2 2024)
- [ ] Video banner generation
- [ ] 3D design capabilities
- [ ] Advanced personalization engine
- [ ] Enterprise SSO integration
- [ ] White-label solutions

### 🌟 Phase 4 - AI Innovation (Q3 2024)
- [ ] Custom model training
- [ ] Voice-to-banner generation
- [ ] Automated campaign optimization
- [ ] Predictive design analytics
- [ ] AR/VR banner experiences

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 100Mbps

### Recommended (Production)
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+
- **Storage**: 500GB+ SSD
- **Network**: 1Gbps+
- **GPU**: NVIDIA RTX 4090 (for T2I models)

### Cloud Deployment
- **AWS**: EC2 g5.2xlarge hoặc higher
- **Google Cloud**: n1-highmem-8 + GPU
- **Azure**: Standard_NC24s_v3

---

## 🎯 Getting Started Checklist

- [ ] Clone repository và setup environment
- [ ] Configure API keys trong .env file
- [ ] Start Redis và database services
- [ ] Run backend API server
- [ ] Start frontend development server
- [ ] Create your first banner via web interface
- [ ] Explore system monitor dashboard
- [ ] Check API documentation
- [ ] Join our Discord community

**🚀 Ready to create amazing banners with AI? Let's get started!**

---

**Multi AI Agent Banner Generator** - Tạo banner quảng cáo chuyên nghiệp với sức mạnh AI tiên tiến