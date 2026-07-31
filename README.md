# Argus: Robotics Pipeline Inspection Copilot

This repository contains the code for the **Argus** agentic pipeline inspection copilot, sold alongside Raphson Robotics' quadruped ("Spider") hardware.

The system features:
1. **CV defect detection** (YOLOv8 segmentation model trained on Severstal Steel Defect Detection dataset)
2. **Gemini reasoning agent** (Severity assessment, maintenance planning, work order generation)
3. **NLP logstore & reports** (ChromaDB vector store + chat query panel)
4. **React frontend dashboard** ( Pastel sky-blue / sea-green design system matching Stitch finalized layout)
5. **FastAPI E2E integration**

## Development Setup

### CV Module Setup
Navigate to the `cv/` directory:
```bash
cd cv
pip install -r requirements.txt
```

### Run Tests
```bash
pytest cv/tests/ -v
```

## Branch Mapping and Flow
All work proceeds sequentially across these branches:
1. `feature/cv-detection` — CV model training and inference contract
2. `feature/agent-reasoning` — Gemini severity & planning agent
3. `feature/reporting-logstore` — NLP report generation & vector search
4. `feature/dashboard-frontend` — React + Tailwind dashboard (Stitch design system)
5. `feature/integration-e2e` — FastAPI orchestrator and e2e integration
