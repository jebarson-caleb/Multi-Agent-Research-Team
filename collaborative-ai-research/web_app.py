"""Minimal web UI for the Collaborative AI Research Team."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.main import ResearchTeam

app = FastAPI(title="Collaborative AI Research Team")
team = ResearchTeam()


class ResearchRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Collaborative AI Research Team</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #111; }
    .container { max-width: 920px; margin: 0 auto; }
    textarea { width: 100%; min-height: 120px; padding: 12px; font-size: 16px; }
    button { padding: 10px 16px; font-size: 16px; cursor: pointer; }
    .result { margin-top: 1.5rem; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
    .row { display: flex; gap: 12px; align-items: center; }
    .meta { color: #444; font-size: 14px; }
    .error { color: #b00020; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Collaborative AI Research Team</h1>
    <p>Enter a research query to run the multi-agent pipeline.</p>
    <textarea id="query" placeholder="e.g., What are the main benefits and risks of LLMs in education?"></textarea>
    <div class="row" style="margin-top: 12px;">
      <button id="run">Run research</button>
      <span id="status" class="meta"></span>
    </div>
    <div id="output" class="result" style="display:none;"></div>
  </div>

  <script>
    const runBtn = document.getElementById('run');
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');

    runBtn.onclick = async () => {
      const query = document.getElementById('query').value.trim();
      if (!query) {
        statusEl.textContent = 'Please enter a query.';
        return;
      }
      statusEl.textContent = 'Running…';
      outputEl.style.display = 'none';
      outputEl.innerHTML = '';

      try {
        const res = await fetch('/api/research', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Request failed');
        }

        const data = await res.json();
        const meta = `Confidence: ${(data.confidence * 100).toFixed(1)}% | ` +
                     `Verification: ${(data.verification_score * 100).toFixed(1)}% | ` +
                     `Tokens: ${data.token_usage} | Cost: $${data.cost.toFixed(4)} | ` +
                     `Duration: ${data.duration.toFixed(2)}s`;

        outputEl.innerHTML = `
          <h2>Summary</h2>
          <p>${data.summary}</p>
          <div class="meta">${meta}</div>
          <h3>Findings</h3>
          <pre>${JSON.stringify(data.detailed_findings, null, 2)}</pre>
        `;
        outputEl.style.display = 'block';
        statusEl.textContent = 'Done.';
      } catch (err) {
        statusEl.textContent = '';
        outputEl.innerHTML = `<p class="error">${err.message}</p>`;
        outputEl.style.display = 'block';
      }
    };
  </script>
</body>
</html>
"""


@app.post("/api/research")
async def api_research(payload: ResearchRequest) -> JSONResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await team.aresearch(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data: dict[str, Any] = asdict(result)
    return JSONResponse(content=data)
