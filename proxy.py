#!/usr/bin/env python3
"""
LLM Proxy Inspector — sits between Claude Code and llama-server,
logs every request/response and serves a live web UI to inspect them.

Usage:
    python3 proxy.py [--port 9001] [--target http://localhost:8001] [--ui-port 9002]

Then set:
    export ANTHROPIC_BASE_URL=http://localhost:9001

Open http://localhost:9002 in a browser to inspect traffic.
"""

import argparse
import json
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# ─── Shared state ───────────────────────────────────────────────────
traffic_log = []
log_lock = threading.Lock()
MAX_LOG_ENTRIES = 200

def add_entry(entry):
    with log_lock:
        traffic_log.append(entry)
        if len(traffic_log) > MAX_LOG_ENTRIES:
            traffic_log.pop(0)

def get_entries_since(idx):
    with log_lock:
        return traffic_log[idx:], len(traffic_log)


# ─── Proxy server ──────────────────────────────────────────────────
class ProxyHandler(BaseHTTPRequestHandler):
    target_base = "http://localhost:8001"

    def log_message(self, fmt, *args):
        pass  # silence default logging

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)

        try:
            body = json.loads(raw_body)
        except:
            body = {"_raw": raw_body.decode("utf-8", errors="replace")}

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Record the request
        add_entry({
            "type": "request",
            "ts": ts,
            "method": "POST",
            "path": self.path,
            "body": body,
        })

        # Forward to target
        target_url = f"{self.target_base}{self.path}"
        req = urllib.request.Request(
            target_url,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req)
            resp_data = resp.read()
            status = resp.status
            resp_content_type = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            resp_data = e.read()
            status = e.code
            resp_content_type = e.headers.get("Content-Type", "application/json")

        # Try to decode response
        resp_text = resp_data.decode("utf-8", errors="replace")

        # Check if it's SSE
        is_sse = "event:" in resp_text[:200]

        add_entry({
            "type": "response",
            "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "status": status,
            "is_sse": is_sse,
            "body": resp_text,
            "size": len(resp_data),
        })

        # Send back to client
        self.send_response(status)
        self.send_header("Content-Type", resp_content_type)
        self.end_headers()
        self.wfile.write(resp_data)

    def do_GET(self):
        # Health check passthrough
        target_url = f"{self.target_base}{self.path}"
        try:
            resp = urllib.request.urlopen(target_url)
            data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "text/plain"))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())


# ─── Web UI server ─────────────────────────────────────────────────
UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Proxy Inspector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3a;
    --text: #e0e0ea;
    --text-dim: #6a6a80;
    --accent: #4af;
    --req-color: #f8a;
    --res-color: #4f8;
    --sse-color: #fa4;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }

  body {
    font-family: var(--mono);
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  header {
    padding: 12px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  header h1 {
    font-family: var(--sans);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--accent);
  }

  header .stats {
    font-size: 11px;
    color: var(--text-dim);
  }

  .controls {
    padding: 8px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 8px;
    align-items: center;
    flex-shrink: 0;
  }

  .controls button {
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface2);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
  }

  .controls button:hover { border-color: var(--accent); color: var(--accent); }
  .controls button.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }

  .controls .spacer { flex: 1; }

  .controls label {
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
  }

  .controls input[type=checkbox] { accent-color: var(--accent); }

  .main-layout {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .entry-list {
    width: 340px;
    min-width: 260px;
    border-right: 1px solid var(--border);
    overflow-y: auto;
    flex-shrink: 0;
  }

  .entry-item {
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.1s;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .entry-item:hover { background: var(--surface2); }
  .entry-item.selected { background: var(--surface2); border-left: 3px solid var(--accent); }

  .entry-item .entry-head {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .entry-item .tag {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 1px 6px;
    border-radius: 3px;
    text-transform: uppercase;
  }

  .tag.req { background: rgba(255,136,170,0.15); color: var(--req-color); }
  .tag.res { background: rgba(68,255,136,0.15); color: var(--res-color); }
  .tag.sse { background: rgba(255,170,68,0.15); color: var(--sse-color); }

  .entry-item .ts {
    font-size: 10px;
    color: var(--text-dim);
  }

  .entry-item .summary {
    font-size: 11px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .detail-pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .detail-header {
    padding: 10px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    flex-shrink: 0;
  }

  .detail-body {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
  }

  .detail-body pre {
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-all;
    color: var(--text);
  }

  .detail-body .section-label {
    font-family: var(--sans);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--accent);
    margin: 16px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }

  .detail-body .section-label:first-child { margin-top: 0; }

  .detail-body .msg-block {
    margin: 8px 0;
    padding: 10px 14px;
    background: var(--surface2);
    border-radius: 6px;
    border-left: 3px solid var(--border);
  }

  .msg-block.system { border-left-color: var(--accent); }
  .msg-block.user { border-left-color: var(--req-color); }
  .msg-block.assistant { border-left-color: var(--res-color); }
  .msg-block.tool_result { border-left-color: var(--sse-color); }
  .msg-block.tool_use { border-left-color: #a8f; }

  .msg-block .role-tag {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }

  .msg-block.system .role-tag { color: var(--accent); }
  .msg-block.user .role-tag { color: var(--req-color); }
  .msg-block.assistant .role-tag { color: var(--res-color); }
  .msg-block.tool_result .role-tag { color: var(--sse-color); }
  .msg-block.tool_use .role-tag { color: #a8f; }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-dim);
    font-size: 13px;
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

  .collapsible-toggle {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    cursor: pointer;
    background: none;
    border: none;
    padding: 2px 0;
  }

  .collapsible-toggle:hover { text-decoration: underline; }

  .sse-event {
    margin: 4px 0;
    padding: 4px 8px;
    background: rgba(255,170,68,0.05);
    border-radius: 3px;
    font-size: 11px;
  }

  .sse-event .event-type {
    color: var(--sse-color);
    font-weight: 700;
  }

  .token-stream {
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.8;
    padding: 10px 14px;
    background: var(--surface2);
    border-radius: 6px;
  }

  .token-stream .token {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 3px;
    padding: 1px 3px;
    margin: 1px;
    display: inline;
    white-space: pre-wrap;
    word-break: break-all;
    color: var(--text);
  }

  .token-stream .tool-token {
    color: #a8f;
    border-color: rgba(170,136,255,0.2);
    background: rgba(170,136,255,0.06);
  }

  .token-stream .err-token {
    color: #f66;
    border-color: rgba(255,100,100,0.2);
  }

  .token-stream .stream-marker {
    display: block;
    margin: 6px 0 4px 0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--sse-color);
  }

  .token-stream .tool-marker {
    color: #a8f;
  }
</style>
</head>
<body>

<header>
  <h1>LLM PROXY INSPECTOR</h1>
  <div class="stats" id="stats">waiting for traffic...</div>
</header>

<div class="controls">
  <button onclick="clearLog()">Clear</button>
  <div class="spacer"></div>
  <label><input type="checkbox" id="autoScroll" checked> Auto-scroll</label>
  <label><input type="checkbox" id="autoSelect" checked> Auto-select latest</label>
  <label><input type="checkbox" id="expandSSE"> Expand SSE events</label>
  <label><input type="checkbox" id="truncateLong" checked> Truncate long content</label>
</div>

<div class="main-layout">
  <div class="entry-list" id="entryList"></div>
  <div class="detail-pane">
    <div class="detail-header" id="detailHeader">Select an entry to inspect</div>
    <div class="detail-body" id="detailBody">
      <div class="empty-state">Traffic will appear here as Claude Code communicates with your local model</div>
    </div>
  </div>
</div>

<script>
const entryList = document.getElementById('entryList');
const detailBody = document.getElementById('detailBody');
const detailHeader = document.getElementById('detailHeader');
const statsEl = document.getElementById('stats');

let entries = [];
let selectedIdx = -1;
let pollIdx = 0;
let reqCount = 0;
let resCount = 0;

function esc(s) {
  if (typeof s !== 'string') s = JSON.stringify(s, null, 2) || '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function truncate(s, max) {
  if (!document.getElementById('truncateLong').checked) return s;
  if (typeof s === 'string' && s.length > max) {
    return s.slice(0, max) + '\n\n... (' + s.length + ' chars total, toggle "Truncate" to see all)';
  }
  return s;
}

function summarize(entry) {
  if (entry.type === 'request') {
    const b = entry.body;
    const msgs = b.messages || [];
    const last = msgs[msgs.length - 1];
    if (last) {
      const c = typeof last.content === 'string' ? last.content : JSON.stringify(last.content);
      return c.slice(0, 80);
    }
    return entry.path;
  } else {
    if (entry.is_sse) return `SSE stream — ${(entry.size/1024).toFixed(1)} KB`;
    return `${entry.status} — ${(entry.size/1024).toFixed(1)} KB`;
  }
}

function renderList() {
  entryList.innerHTML = '';
  entries.forEach((e, i) => {
    const div = document.createElement('div');
    div.className = 'entry-item' + (i === selectedIdx ? ' selected' : '');
    const isReq = e.type === 'request';
    const tagClass = isReq ? 'req' : (e.is_sse ? 'sse' : 'res');
    const tagText = isReq ? 'REQ' : (e.is_sse ? 'SSE' : 'RES');
    div.innerHTML = `
      <div class="entry-head">
        <span class="tag ${tagClass}">${tagText}</span>
        <span class="ts">${esc(e.ts)}</span>
      </div>
      <div class="summary">${esc(summarize(e))}</div>
    `;
    div.onclick = () => selectEntry(i);
    entryList.appendChild(div);
  });

  if (document.getElementById('autoScroll').checked) {
    entryList.scrollTop = entryList.scrollHeight;
  }
}

function renderRequestDetail(entry) {
  const b = entry.body;
  let html = '';

  html += `<div class="section-label">Endpoint</div>`;
  html += `<pre>POST ${esc(entry.path)}</pre>`;

  // System prompt
  if (b.system) {
    html += `<div class="section-label">System Prompt</div>`;
    const sys = typeof b.system === 'string' ? b.system : JSON.stringify(b.system, null, 2);
    const charCount = sys.length;
    html += `<div class="msg-block system">`;
    html += `<div class="role-tag">SYSTEM (${charCount.toLocaleString()} chars)</div>`;
    html += `<pre>${esc(truncate(sys, 8000))}</pre>`;
    html += `</div>`;
  }

  // Messages
  if (b.messages) {
    html += `<div class="section-label">Messages (${b.messages.length})</div>`;
    b.messages.forEach((m, i) => {
      const role = m.role || 'unknown';
      let content = m.content;

      // Handle array content (tool use, tool results, etc.)
      if (Array.isArray(content)) {
        content.forEach(block => {
          const blockType = block.type || role;
          const cssClass = blockType === 'tool_use' ? 'tool_use' :
                           blockType === 'tool_result' ? 'tool_result' : role;
          html += `<div class="msg-block ${cssClass}">`;

          if (block.type === 'tool_use') {
            html += `<div class="role-tag">TOOL CALL: ${esc(block.name || '?')}</div>`;
            html += `<pre>${esc(truncate(JSON.stringify(block.input || block, null, 2), 4000))}</pre>`;
          } else if (block.type === 'tool_result') {
            html += `<div class="role-tag">TOOL RESULT (${esc(block.tool_use_id || '')})</div>`;
            const rc = typeof block.content === 'string' ? block.content : JSON.stringify(block.content, null, 2);
            html += `<pre>${esc(truncate(rc, 4000))}</pre>`;
          } else if (block.type === 'text') {
            html += `<div class="role-tag">${esc(role.toUpperCase())}</div>`;
            html += `<pre>${esc(truncate(block.text || '', 4000))}</pre>`;
          } else {
            html += `<div class="role-tag">${esc(role.toUpperCase())}</div>`;
            html += `<pre>${esc(truncate(JSON.stringify(block, null, 2), 4000))}</pre>`;
          }
          html += `</div>`;
        });
      } else if (typeof content === 'object' && content !== null && content.type === 'tool_result') {
        html += `<div class="msg-block tool_result">`;
        html += `<div class="role-tag">TOOL RESULT</div>`;
        html += `<pre>${esc(truncate(JSON.stringify(content, null, 2), 4000))}</pre>`;
        html += `</div>`;
      } else {
        // Plain string content
        const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
        html += `<div class="msg-block ${role}">`;
        html += `<div class="role-tag">${esc(role.toUpperCase())} (${text.length.toLocaleString()} chars)</div>`;
        html += `<pre>${esc(truncate(text, 4000))}</pre>`;
        html += `</div>`;
      }
    });
  }

  // Tools
  if (b.tools) {
    html += `<div class="section-label">Tools (${b.tools.length})</div>`;
    html += `<pre>${esc(truncate(JSON.stringify(b.tools, null, 2), 3000))}</pre>`;
  }

  // Other fields
  const skip = new Set(['system', 'messages', 'tools']);
  const other = Object.keys(b).filter(k => !skip.has(k));
  if (other.length) {
    html += `<div class="section-label">Other Fields</div>`;
    const obj = {};
    other.forEach(k => obj[k] = b[k]);
    html += `<pre>${esc(JSON.stringify(obj, null, 2))}</pre>`;
  }

  return html;
}

function renderResponseDetail(entry) {
  let html = '';
  html += `<div class="section-label">Response — ${entry.status} — ${(entry.size/1024).toFixed(1)} KB</div>`;

  if (entry.is_sse) {
    const expandSSE = document.getElementById('expandSSE').checked;
    const lines = entry.body.split('\n');
    let events = [];
    let current = null;

    for (const line of lines) {
      if (line.startsWith('event:')) {
        if (current) events.push(current);
        current = { event: line.slice(6).trim(), data: '' };
      } else if (line.startsWith('data:') && current) {
        current.data += line.slice(5).trim();
      } else if (line === '' && current) {
        events.push(current);
        current = null;
      }
    }
    if (current) events.push(current);

    // Assemble the full text from text_delta events
    let fullText = '';
    events.forEach(ev => {
      try {
        const d = JSON.parse(ev.data);
        if (d.delta && d.delta.text) fullText += d.delta.text;
        if (d.delta && d.delta.partial_json) fullText += d.delta.partial_json;
      } catch {}
    });

    if (fullText) {
      html += `<div class="section-label">Assembled Output</div>`;
      html += `<div class="msg-block assistant">`;
      html += `<div class="role-tag">ASSISTANT (${fullText.length.toLocaleString()} chars)</div>`;
      html += `<pre>${esc(truncate(fullText, 8000))}</pre>`;
      html += `</div>`;
    }

    html += `<div class="section-label">SSE Events (${events.length})</div>`;
    if (expandSSE) {
      // Compact token stream: show tokens inline, structural events as labels
      html += `<div class="token-stream">`;
      events.forEach(ev => {
        try {
          const d = JSON.parse(ev.data);
          if (d.delta && d.delta.text) {
            // Text token — show inline
            html += `<span class="token">${esc(d.delta.text)}</span>`;
          } else if (d.delta && d.delta.partial_json) {
            // Tool call JSON fragment
            html += `<span class="token tool-token">${esc(d.delta.partial_json)}</span>`;
          } else if (ev.event === 'content_block_start') {
            const block = d.content_block || {};
            if (block.type === 'tool_use') {
              html += `<div class="stream-marker tool-marker">▶ tool_use: ${esc(block.name || '?')}</div>`;
            } else {
              html += `<div class="stream-marker">▶ ${esc(block.type || ev.event)}</div>`;
            }
          } else if (ev.event === 'content_block_stop') {
            html += `<div class="stream-marker">■ block_stop</div>`;
          } else if (ev.event === 'message_start') {
            const u = d.message && d.message.usage;
            if (u) {
              html += `<div class="stream-marker">● message_start (input: ${u.input_tokens} tokens)</div>`;
            }
          } else if (ev.event === 'message_delta') {
            const sr = d.delta && d.delta.stop_reason;
            const u = d.usage;
            let info = sr ? `stop: ${sr}` : '';
            if (u && u.output_tokens) info += ` (output: ${u.output_tokens} tokens)`;
            html += `<div class="stream-marker">● message_delta ${info}</div>`;
          } else if (ev.event !== 'content_block_delta') {
            // Other structural events
            html += `<div class="stream-marker">${esc(ev.event)}</div>`;
          }
        } catch {
          html += `<span class="token err-token">${esc(ev.data.slice(0, 60))}</span>`;
        }
      });
      html += `</div>`;
    } else {
      html += `<pre style="color:var(--text-dim)">Toggle "Expand SSE events" to see token stream</pre>`;
    }
  } else {
    // JSON response
    try {
      const parsed = JSON.parse(entry.body);
      html += `<pre>${esc(truncate(JSON.stringify(parsed, null, 2), 8000))}</pre>`;
    } catch {
      html += `<pre>${esc(truncate(entry.body, 8000))}</pre>`;
    }
  }

  return html;
}

function selectEntry(i) {
  selectedIdx = i;
  const entry = entries[i];
  if (!entry) return;

  const isReq = entry.type === 'request';
  detailHeader.textContent = `${isReq ? 'REQUEST' : 'RESPONSE'} @ ${entry.ts}`;

  if (isReq) {
    detailBody.innerHTML = renderRequestDetail(entry);
  } else {
    detailBody.innerHTML = renderResponseDetail(entry);
  }

  // Update selected state in list
  document.querySelectorAll('.entry-item').forEach((el, j) => {
    el.classList.toggle('selected', j === i);
  });
}

function clearLog() {
  fetch('/api/clear', { method: 'POST' });
  entries = [];
  pollIdx = 0;
  reqCount = 0;
  resCount = 0;
  selectedIdx = -1;
  renderList();
  detailBody.innerHTML = '<div class="empty-state">Cleared</div>';
  detailHeader.textContent = 'Select an entry to inspect';
  statsEl.textContent = 'waiting for traffic...';
}

async function poll() {
  try {
    const resp = await fetch(`/api/entries?since=${pollIdx}`);
    const data = await resp.json();
    if (data.entries.length > 0) {
      entries.push(...data.entries);
      pollIdx = data.total;
      data.entries.forEach(e => {
        if (e.type === 'request') reqCount++;
        else resCount++;
      });
      renderList();
      statsEl.textContent = `${reqCount} requests, ${resCount} responses`;
      if (document.getElementById('autoSelect').checked) {
        selectEntry(entries.length - 1);
      }
    }
  } catch {}
  setTimeout(poll, 500);
}

poll();
</script>
</body>
</html>"""


class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(UI_HTML.encode())
        elif self.path.startswith('/api/entries'):
            since = 0
            if '?' in self.path:
                for part in self.path.split('?')[1].split('&'):
                    if part.startswith('since='):
                        since = int(part.split('=')[1])
            new_entries, total = get_entries_since(since)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"entries": new_entries, "total": total}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/clear':
            with log_lock:
                traffic_log.clear()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


# ─── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM Proxy Inspector")
    parser.add_argument("--port", type=int, default=9001, help="Proxy port (default: 9001)")
    parser.add_argument("--target", type=str, default="http://localhost:8001", help="Target llama-server URL")
    parser.add_argument("--ui-port", type=int, default=9002, help="Web UI port (default: 9002)")
    args = parser.parse_args()

    ProxyHandler.target_base = args.target

    proxy_server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    ui_server = HTTPServer(("127.0.0.1", args.ui_port), UIHandler)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║          LLM Proxy Inspector                    ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Proxy:    http://localhost:{args.port:<5}               ║")
    print(f"║  Target:   {args.target:<38}║")
    print(f"║  Web UI:   http://localhost:{args.ui_port:<5}               ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Set: export ANTHROPIC_BASE_URL=http://localhost:{args.port}  ║")
    print(f"╚══════════════════════════════════════════════════╝")

    threading.Thread(target=proxy_server.serve_forever, daemon=True).start()
    threading.Thread(target=ui_server.serve_forever, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        proxy_server.shutdown()
        ui_server.shutdown()


if __name__ == "__main__":
    main()
