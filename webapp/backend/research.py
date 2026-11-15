"""Read-only snapshot of recorded experiments for the separate research website."""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix='/api/research')
ASSETS = Path(__file__).with_name('research_assets')


@router.get('/benchmark')
def benchmark():
    path = ASSETS / 'benchmark.json'
    if not path.exists():
        raise HTTPException(503, 'Benchmark evidence has not been generated yet.')
    return json.loads(path.read_text(encoding='utf-8'))


@router.get('/results')
def results():
    return {key: json.loads((ASSETS/name).read_text(encoding='utf-8')) for key,name in
            [('evaluation','results.json'),('quality','quality.json'),('decision','decision.json'),('paired','paired.json')]}


@router.get('/figures/{name}')
def figure(name: str):
    if name not in {'synthetic_comparison.png','quality_metrics.png'}:
        raise HTTPException(404, 'Unknown research figure')
    return FileResponse(ASSETS/name, media_type='image/png')
