/**
 * Agentic RAG — AI Research Workspace Client Controller
 * 
 * Features:
 * - Editorial light-mode enterprise interaction patterns
 * - Server-Sent Events (SSE) streaming (/query/stream) with requestAnimationFrame batching
 * - Deterministic 4-stage pipeline (Planning, Retrieving, Reranking, Synthesizing)
 * - Research Source Inspector drawer with citation deep-linking & focus management
 * - Full keyboard accessibility (ESC, focus trap, visible focus, ARIA live announcements)
 * - Session memory management & visual reset states (/memory/clear)
 * - Placeholder-shielded markdown rendering (code blocks immune to citation regex corruption)
 * - Streaming-aware code block detection & Windows CRLF compatibility
 * - Desktop sidebar collapse (Ctrl+B) and mobile drawer
 * - AbortController stop response handling
 */

(function () {
  'use strict';

  // ── State Management ────────────────────────────────────────────────────────
  let currentSessionId = getOrCreateSessionId();
  let currentAbortController = null;
  let isGenerating = false;
  let activeSources = [];
  let lastInspectorTrigger = null;

  // ── DOM References ─────────────────────────────────────────────────────────
  const appLayout = document.querySelector('.app-layout');
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');
  const mobileToggleBtn = document.getElementById('mobileToggleBtn');
  const mobileCloseBtn = document.getElementById('mobileCloseBtn');
  const desktopToggleBtn = document.getElementById('desktopToggleBtn');

  const memoryIdDisplay = document.getElementById('memoryIdDisplay');
  const copyMemoryIdBtn = document.getElementById('copyMemoryIdBtn');
  const clearMemoryBtn = document.getElementById('clearMemoryBtn');
  const serviceStatusText = document.getElementById('serviceStatusText');

  const messagesContainer = document.getElementById('messagesContainer');
  const heroWelcome = document.getElementById('heroWelcome');
  const chatThread = document.getElementById('chatThread');

  const promptForm = document.getElementById('promptForm');
  const promptInput = document.getElementById('promptInput');
  const sendBtn = document.getElementById('sendBtn');

  const generationBar = document.getElementById('generationBar');
  const genStatusText = document.getElementById('genStatusText');
  const stopResponseBtn = document.getElementById('stopResponseBtn');

  const sourceInspector = document.getElementById('sourceInspector');
  const inspectorBackdrop = document.getElementById('inspectorBackdrop');
  const closeInspectorBtn = document.getElementById('closeInspectorBtn');
  const topbarInspectorBtn = document.getElementById('topbarInspectorBtn');
  const topbarSourceCount = document.getElementById('topbarSourceCount');
  const topbarStatusText = document.getElementById('topbarStatusText');
  const topbarChipDot = document.getElementById('topbarChipDot');
  const inspectorEmpty = document.getElementById('inspectorEmpty');
  const inspectorSourcesList = document.getElementById('inspectorSourcesList');
  const inspectorStats = document.getElementById('inspectorStats');

  const toast = document.getElementById('toast');
  const srAnnouncer = document.getElementById('srAnnouncer');

  // ── Initialization ─────────────────────────────────────────────────────────
  function init() {
    updateMemoryDisplay();
    setupEventListeners();
    setupTextareaAutoResize();
    checkServiceHealth();
  }

  function getOrCreateSessionId() {
    let id = localStorage.getItem('rag_session_id');
    if (!id) {
      id = (typeof crypto.randomUUID === 'function') 
        ? crypto.randomUUID() 
        : 'sess_' + Math.random().toString(36).substring(2, 12);
      localStorage.setItem('rag_session_id', id);
    }
    return id;
  }

  function updateMemoryDisplay() {
    if (memoryIdDisplay) {
      memoryIdDisplay.textContent = currentSessionId.substring(0, 8);
    }
  }

  function announceStatus(message) {
    if (srAnnouncer) {
      srAnnouncer.textContent = message;
    }
  }

  async function checkServiceHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        const isOnline = data.status === 'online';
        if (serviceStatusText) {
          serviceStatusText.textContent = isOnline ? 'Online' : 'Degraded';
        }
        if (topbarStatusText) {
          topbarStatusText.textContent = isOnline ? 'Ready' : 'Degraded';
        }
        if (topbarChipDot) {
          topbarChipDot.className = 'chip-dot' + (isOnline ? '' : ' warning');
        }
      } else {
        markServiceDegraded();
      }
    } catch {
      markServiceDegraded();
    }
  }

  function markServiceDegraded() {
    if (serviceStatusText) serviceStatusText.textContent = 'Standby';
    if (topbarStatusText) topbarStatusText.textContent = 'Standby';
    if (topbarChipDot) topbarChipDot.className = 'chip-dot standby';
  }

  // ── Event Listeners ────────────────────────────────────────────────────────
  function setupEventListeners() {
    // Desktop Sidebar Collapse Toggle
    if (desktopToggleBtn) {
      desktopToggleBtn.addEventListener('click', toggleDesktopSidebar);
    }

    // Mobile Sidebar
    if (mobileToggleBtn) {
      mobileToggleBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        sidebarBackdrop.classList.add('open');
        mobileCloseBtn && mobileCloseBtn.focus();
      });
    }

    if (mobileCloseBtn) {
      mobileCloseBtn.addEventListener('click', closeMobileSidebar);
    }

    if (sidebarBackdrop) {
      sidebarBackdrop.addEventListener('click', closeMobileSidebar);
    }

    // Source Inspector Drawer
    if (topbarInspectorBtn) {
      topbarInspectorBtn.addEventListener('click', (e) => {
        if (sourceInspector.classList.contains('open')) {
          closeSourceInspector();
        } else {
          lastInspectorTrigger = topbarInspectorBtn;
          openSourceInspector();
        }
      });
    }

    if (closeInspectorBtn) {
      closeInspectorBtn.addEventListener('click', closeSourceInspector);
    }

    if (inspectorBackdrop) {
      inspectorBackdrop.addEventListener('click', closeSourceInspector);
    }

    // Global Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
      // Escape closes open overlays
      if (e.key === 'Escape') {
        if (sourceInspector.classList.contains('open')) {
          closeSourceInspector();
        }
        if (sidebar.classList.contains('open')) {
          closeMobileSidebar();
        }
      }

      // Ctrl+B / Cmd+B toggles sidebar
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleDesktopSidebar();
      }

      // Trap focus in Source Inspector when open
      if (sourceInspector.classList.contains('open') && e.key === 'Tab') {
        trapFocusInInspector(e);
      }
    });

    // Copy Session ID
    if (copyMemoryIdBtn) {
      copyMemoryIdBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(currentSessionId).then(() => {
          showToast('Session ID copied to clipboard');
        }).catch(() => {
          showToast('Failed to copy ID');
        });
      });
    }

    // Clear Memory
    if (clearMemoryBtn) {
      clearMemoryBtn.addEventListener('click', handleClearMemory);
    }

    // Stop Response
    if (stopResponseBtn) {
      stopResponseBtn.addEventListener('click', handleStopResponse);
    }

    // Prompt Form Submit
    if (promptForm) {
      promptForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSubmit();
      });
    }

    // Enter to Send, Shift+Enter for Newline
    if (promptInput) {
      promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          handleSubmit();
        }
      });
    }

    // Suggested Inquiries & Hero Chips
    document.querySelectorAll('.suggestion-item, .hero-chip').forEach(el => {
      el.addEventListener('click', () => {
        const prompt = el.getAttribute('data-prompt');
        if (prompt && !isGenerating) {
          promptInput.value = prompt;
          promptInput.style.height = 'auto';
          promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + 'px';
          sendBtn.disabled = false;
          handleSubmit();
          closeMobileSidebar();
        }
      });
    });
  }

  function toggleDesktopSidebar() {
    if (sidebar && appLayout) {
      const isCollapsed = sidebar.classList.toggle('collapsed');
      appLayout.classList.toggle('sidebar-collapsed', isCollapsed);
      if (desktopToggleBtn) {
        desktopToggleBtn.setAttribute('aria-expanded', String(!isCollapsed));
      }
    }
  }

  function closeMobileSidebar() {
    sidebar.classList.remove('open');
    sidebarBackdrop.classList.remove('open');
    if (mobileToggleBtn) mobileToggleBtn.focus();
  }

  function setupTextareaAutoResize() {
    promptInput.addEventListener('input', () => {
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + 'px';
      sendBtn.disabled = promptInput.value.trim() === '' || isGenerating;
    });
  }

  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2500);
  }

  // ── Source Inspector Drawer Controller ─────────────────────────────────────
  function openSourceInspector(sourceIndexToHighlight = null) {
    sourceInspector.classList.add('open');
    inspectorBackdrop.classList.add('open');
    sourceInspector.setAttribute('aria-hidden', 'false');

    if (sourceIndexToHighlight !== null) {
      const targetCard = document.getElementById(`inspector-card-${sourceIndexToHighlight}`);
      if (targetCard) {
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        targetCard.classList.add('is-highlighted');
        setTimeout(() => targetCard.classList.remove('is-highlighted'), 2400);
      }
    }

    // Accessible focus management
    if (closeInspectorBtn) {
      closeInspectorBtn.focus();
    }
  }

  function closeSourceInspector() {
    sourceInspector.classList.remove('open');
    inspectorBackdrop.classList.remove('open');
    sourceInspector.setAttribute('aria-hidden', 'true');

    // Restore focus to element that triggered inspector
    if (lastInspectorTrigger && typeof lastInspectorTrigger.focus === 'function') {
      lastInspectorTrigger.focus();
    }
  }

  function trapFocusInInspector(e) {
    const focusable = sourceInspector.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  function updateInspectorView(sources) {
    activeSources = sources || [];
    const count = activeSources.length;

    if (topbarSourceCount) {
      topbarSourceCount.textContent = String(count);
    }
    if (topbarInspectorBtn) {
      if (count > 0) {
        topbarInspectorBtn.classList.add('has-sources');
      } else {
        topbarInspectorBtn.classList.remove('has-sources');
      }
    }

    if (inspectorStats) {
      inspectorStats.textContent = `${count} source passage${count === 1 ? '' : 's'} retrieved`;
    }

    if (count === 0) {
      if (inspectorEmpty) inspectorEmpty.style.display = 'block';
      if (inspectorSourcesList) {
        inspectorSourcesList.style.display = 'none';
        inspectorSourcesList.innerHTML = '';
      }
      return;
    }

    if (inspectorEmpty) inspectorEmpty.style.display = 'none';
    if (inspectorSourcesList) {
      inspectorSourcesList.style.display = 'flex';
      inspectorSourcesList.innerHTML = '';

      activeSources.forEach((s, idx) => {
        const itemIdx = idx + 1;
        const card = document.createElement('div');
        card.className = 'inspector-source-card';
        card.id = `inspector-card-${itemIdx}`;

        const scoreText = (s.score !== null && s.score !== undefined)
          ? `Score: ${typeof s.score === 'number' ? s.score.toFixed(3) : s.score}`
          : 'Retrieved passage';

        card.innerHTML = `
          <div class="source-card-header">
            <div class="source-badge-group">
              <span class="source-index-tag">[${itemIdx}]</span>
              <span class="source-filename" title="${escapeHtml(s.source || 'Document')}">${escapeHtml(s.source || 'Document')}</span>
            </div>
            <span class="source-score-badge">${scoreText}</span>
          </div>
          <div class="source-snippet">${escapeHtml(s.content || '')}</div>
          <div class="source-card-footer">
            <button type="button" class="btn-copy-snippet" data-copy-target="${itemIdx}" aria-label="Copy passage ${itemIdx}">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              <span>Copy passage</span>
            </button>
          </div>
        `;

        const copyBtn = card.querySelector('.btn-copy-snippet');
        copyBtn.addEventListener('click', () => {
          navigator.clipboard.writeText(s.content || '').then(() => {
            copyBtn.innerHTML = `
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#32765B" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
              <span style="color:#32765B;">Copied</span>
            `;
            setTimeout(() => {
              copyBtn.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <span>Copy passage</span>
              `;
            }, 1800);
          });
        });

        inspectorSourcesList.appendChild(card);
      });
    }
  }

  // ── Memory Clear Handler ───────────────────────────────────────────────────
  async function handleClearMemory() {
    if (isGenerating) {
      handleStopResponse();
    }

    if (clearMemoryBtn) {
      clearMemoryBtn.disabled = true;
      clearMemoryBtn.classList.add('is-clearing');
      clearMemoryBtn.innerHTML = `
        <span class="btn-spinner" aria-hidden="true"></span>
        <span>Resetting...</span>
      `;
    }

    try {
      await fetch('/memory/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: currentSessionId })
      });
    } catch (e) {
      console.warn('Memory reset signal emitted:', e);
    } finally {
      if (clearMemoryBtn) {
        clearMemoryBtn.disabled = false;
        clearMemoryBtn.classList.remove('is-clearing');
        clearMemoryBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          <span>Reset Session Memory</span>
        `;
      }
    }

    // Rotate to fresh session
    currentSessionId = (typeof crypto.randomUUID === 'function') 
      ? crypto.randomUUID() 
      : 'sess_' + Math.random().toString(36).substring(2, 12);
    localStorage.setItem('rag_session_id', currentSessionId);
    updateMemoryDisplay();

    // Clear UI state
    chatThread.innerHTML = '';
    heroWelcome.style.display = 'block';
    updateInspectorView([]);
    closeSourceInspector();
    announceStatus('Session history and memory cleared');
    showToast('Session history and memory cleared');
  }

  // ── Stop Response Handler ──────────────────────────────────────────────────
  function handleStopResponse() {
    if (currentAbortController && isGenerating) {
      currentAbortController.abort();
      currentAbortController = null;
      setGenerationActive(false);
      showToast('Generation halted by user');
      announceStatus('Response generation halted by user');

      const cursors = document.querySelectorAll('.typing-cursor');
      cursors.forEach(c => c.remove());
    }
  }

  function setGenerationActive(active, statusText = '') {
    isGenerating = active;
    generationBar.style.display = active ? 'flex' : 'none';
    sendBtn.disabled = active || promptInput.value.trim() === '';

    if (active) {
      genStatusText.textContent = statusText || 'Agent is reasoning & synthesizing...';
    }
  }

  // ── Query Submission & SSE Streaming ──────────────────────────────────────
  async function handleSubmit() {
    const query = promptInput.value.trim();
    if (!query || isGenerating) return;

    // Reset prompt input
    promptInput.value = '';
    promptInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Hide hero state
    if (heroWelcome.style.display !== 'none') {
      heroWelcome.style.display = 'none';
    }

    // Append User Message
    appendUserMessage(query);

    // Create Assistant Response Component
    const assistantCard = createAssistantCard();
    chatThread.appendChild(assistantCard.container);
    scrollToBottom();

    // AbortController
    currentAbortController = new AbortController();
    setGenerationActive(true, 'Evaluating intent & guardrails...');
    announceStatus('Evaluating intent and guardrails');

    let accumulatedMarkdown = '';
    const turnSources = [];
    let streamFinalized = false;
    let renderScheduled = false;

    // Throttled render function using requestAnimationFrame for smooth 60fps streaming
    function scheduleRender() {
      if (renderScheduled) return;
      renderScheduled = true;
      requestAnimationFrame(() => {
        assistantCard.updateAnswer(accumulatedMarkdown);
        scrollToBottom();
        renderScheduled = false;
      });
    }

    try {
      const response = await fetch('/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: query, thread_id: currentSessionId }),
        signal: currentAbortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function processEventLine(rawBlock) {
        const trimmed = rawBlock.trim();
        if (!trimmed) return;

        // Split by lines in case multiple lines exist in block
        const blockLines = trimmed.split(/\r?\n/);
        for (const line of blockLines) {
          const lineTrimmed = line.trim();
          if (!lineTrimmed.startsWith('data:')) continue;

          const jsonStr = lineTrimmed.replace(/^data:\s*/, '');
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'thought') {
              assistantCard.addThought(data.step);
              genStatusText.textContent = data.step;
              scrollToBottom();
            } else if (data.type === 'sources') {
              if (Array.isArray(data.sources)) {
                turnSources.push(...data.sources);
                assistantCard.attachSources(turnSources);
                updateInspectorView(turnSources);
                announceStatus(`Retrieved ${turnSources.length} sources`);
              }
            } else if (data.type === 'token') {
              accumulatedMarkdown += data.content;
              scheduleRender();
            } else if (data.type === 'done') {
              // Flush final render before finalizing
              assistantCard.updateAnswer(accumulatedMarkdown);
              assistantCard.finalize(data.status);
              streamFinalized = true;
              setGenerationActive(false);
              announceStatus(`Synthesis complete: ${data.status}`);
            }
          } catch (jsonErr) {
            console.warn('Malformed SSE payload:', jsonStr, jsonErr);
          }
        }
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // Handle both CRLF and LF double-newlines
        const parts = buffer.split(/(?:\r?\n){2}/);
        buffer = parts.pop() || '';

        for (const part of parts) {
          processEventLine(part);
        }
      }

      // Flush any trailing event in buffer
      if (buffer.trim()) {
        processEventLine(buffer);
      }

      // Ensure final render is applied
      assistantCard.updateAnswer(accumulatedMarkdown);

      // Only finalize if done event was not already processed
      if (!streamFinalized) {
        assistantCard.finalize('complete');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        assistantCard.markHalted();
      } else {
        console.error('Streaming request error:', err);
        assistantCard.markError(err.message || 'Network connection failed');
      }
    } finally {
      currentAbortController = null;
      setGenerationActive(false);
      promptInput.focus();
    }
  }

  // ── Message Renderers ──────────────────────────────────────────────────────
  function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message-user';
    msg.textContent = text;
    chatThread.appendChild(msg);
  }

  function createAssistantCard() {
    const container = document.createElement('div');
    container.className = 'message-assistant';

    // 1. Meta Bar
    const metaBar = document.createElement('div');
    metaBar.className = 'assistant-meta-bar';
    metaBar.innerHTML = `
      <div class="assistant-title-group">
        <div class="assistant-glyph" aria-hidden="true">✦</div>
        <span class="assistant-label">Agentic RAG Synthesis</span>
      </div>
    `;
    container.appendChild(metaBar);

    // 2. Agent Execution Trace (Section 4.G)
    const traceContainer = document.createElement('div');
    traceContainer.className = 'agent-trace-container';
    traceContainer.innerHTML = `
      <div class="trace-summary" role="button" tabindex="0" aria-expanded="false" title="Click to view backend execution trace">
        <div class="trace-stages-track">
          <div class="trace-stage is-active" data-stage="planning">
            <span class="stage-dot"></span>
            <span class="stage-label">Planning</span>
          </div>
          <span class="trace-stage-sep" aria-hidden="true">→</span>
          <div class="trace-stage" data-stage="retrieving">
            <span class="stage-dot"></span>
            <span class="stage-label">Retrieving</span>
          </div>
          <span class="trace-stage-sep" aria-hidden="true">→</span>
          <div class="trace-stage" data-stage="reranking">
            <span class="stage-dot"></span>
            <span class="stage-label">Reranking</span>
          </div>
          <span class="trace-stage-sep" aria-hidden="true">→</span>
          <div class="trace-stage" data-stage="synthesizing">
            <span class="stage-dot"></span>
            <span class="stage-label">Synthesizing</span>
          </div>
        </div>
        <div class="trace-meta">
          <span class="trace-status-pill active">Running</span>
          <svg class="trace-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
        </div>
      </div>
      <div class="trace-drawer">
        <div class="trace-log-items"></div>
      </div>
    `;

    const traceSummary = traceContainer.querySelector('.trace-summary');
    const traceStatusPill = traceContainer.querySelector('.trace-status-pill');
    const traceLogItems = traceContainer.querySelector('.trace-log-items');
    const stageNodes = {
      planning: traceContainer.querySelector('[data-stage="planning"]'),
      retrieving: traceContainer.querySelector('[data-stage="retrieving"]'),
      reranking: traceContainer.querySelector('[data-stage="reranking"]'),
      synthesizing: traceContainer.querySelector('[data-stage="synthesizing"]')
    };

    function toggleTrace() {
      const isOpen = traceContainer.classList.toggle('open');
      traceSummary.setAttribute('aria-expanded', String(isOpen));
    }

    traceSummary.addEventListener('click', toggleTrace);
    traceSummary.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggleTrace();
      }
    });

    container.appendChild(traceContainer);

    // 3. Assistant Body with Signature Evidence Rule
    const bodyContainer = document.createElement('div');
    bodyContainer.className = 'assistant-body';

    const answerContent = document.createElement('div');
    answerContent.className = 'answer-content';
    answerContent.innerHTML = '<span class="typing-cursor">▌</span>';
    bodyContainer.appendChild(answerContent);

    // 4. Sources Action Bar (initially hidden)
    const sourcesBar = document.createElement('div');
    sourcesBar.className = 'message-sources-bar';
    sourcesBar.style.display = 'none';
    bodyContainer.appendChild(sourcesBar);

    container.appendChild(bodyContainer);

    let currentActiveStage = 'planning';
    let localSources = [];
    let isGuardrailsBlocked = false;

    // Transition stages cleanly without falsely completing unreached stages
    function setStage(stageName) {
      if (currentActiveStage === stageName) return;

      const order = ['planning', 'retrieving', 'reranking', 'synthesizing'];
      const prevIdx = order.indexOf(currentActiveStage);
      const newIdx = order.indexOf(stageName);

      if (newIdx > prevIdx) {
        for (let i = 0; i < newIdx; i++) {
          const st = stageNodes[order[i]];
          // Only mark as complete if NOT skipped
          if (st && !st.classList.contains('is-skipped') && !st.classList.contains('is-blocked')) {
            st.classList.remove('is-active');
            st.classList.add('is-complete');
          }
        }
      }

      currentActiveStage = stageName;
      const target = stageNodes[stageName];
      if (target && !target.classList.contains('is-skipped') && !target.classList.contains('is-blocked')) {
        target.classList.remove('is-complete');
        target.classList.add('is-active');
      }
    }

    return {
      container,
      addThought(step) {
        const item = document.createElement('div');
        item.className = 'trace-log-item';
        item.innerHTML = `<span class="trace-log-bullet">↳</span><span>${escapeHtml(step)}</span>`;
        traceLogItems.appendChild(item);

        const lower = step.toLowerCase();

        // Check for safety guardrail block
        if (lower.includes('guardrails filter triggered') || lower.includes('safety gate triggered') || (lower.includes('blocked') && lower.includes('guardrail'))) {
          isGuardrailsBlocked = true;
          if (stageNodes.planning) {
            stageNodes.planning.classList.remove('is-active', 'is-complete');
            stageNodes.planning.classList.add('is-blocked');
          }
          if (stageNodes.retrieving) stageNodes.retrieving.classList.add('is-skipped');
          if (stageNodes.reranking) stageNodes.reranking.classList.add('is-skipped');
          if (stageNodes.synthesizing) stageNodes.synthesizing.classList.add('is-skipped');
          traceStatusPill.className = 'trace-status-pill blocked';
          traceStatusPill.textContent = 'Guardrails Blocked';
          return;
        }

        // Check for conversational / greeting bypass
        if (lower.includes('conversational') || lower.includes('greeting') || lower.includes('dialog flow')) {
          if (stageNodes.planning) {
            stageNodes.planning.classList.remove('is-active');
            stageNodes.planning.classList.add('is-complete');
          }
          // Mark vector search and reranking as skipped
          if (stageNodes.retrieving) {
            stageNodes.retrieving.classList.remove('is-active', 'is-complete');
            stageNodes.retrieving.classList.add('is-skipped');
          }
          if (stageNodes.reranking) {
            stageNodes.reranking.classList.remove('is-active', 'is-complete');
            stageNodes.reranking.classList.add('is-skipped');
          }
          currentActiveStage = 'synthesizing';
          if (stageNodes.synthesizing) {
            stageNodes.synthesizing.classList.add('is-active');
          }
          return;
        }

        // Standard stage progression for technical queries
        if (lower.includes('guardrail') || lower.includes('planner') || lower.includes('evaluating intent')) {
          setStage('planning');
        } else if (lower.includes('searching') || lower.includes('qdrant') || (lower.includes('retriev') && !lower.includes('flashrank'))) {
          setStage('retrieving');
        } else if (lower.includes('flashrank') || lower.includes('cross-encoder') || lower.includes('rerank')) {
          setStage('reranking');
        } else if (lower.includes('synthesizing')) {
          setStage('synthesizing');
        }
      },

      updateAnswer(mdText) {
        if (!isGuardrailsBlocked) {
          setStage('synthesizing');
        }
        answerContent.innerHTML = renderMarkdown(mdText) + '<span class="typing-cursor">▌</span>';
        bindCodeCopyButtons(answerContent);
        bindCitationBadges(answerContent, localSources);
      },

      attachSources(sources) {
        localSources = sources || [];
        if (localSources.length === 0) return;

        sourcesBar.style.display = 'flex';
        sourcesBar.innerHTML = `
          <button type="button" class="btn-inspect-context" aria-label="Inspect ${localSources.length} retrieved sources">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
            <span>Inspecting <strong>${localSources.length}</strong> Retrieved Sources</span>
          </button>
        `;

        const inspectBtn = sourcesBar.querySelector('.btn-inspect-context');
        inspectBtn.addEventListener('click', () => {
          lastInspectorTrigger = inspectBtn;
          updateInspectorView(localSources);
          openSourceInspector();
        });

        // Re-bind citation buttons to pass updated sources
        bindCitationBadges(answerContent, localSources);
      },

      finalize(status) {
        const cursor = answerContent.querySelector('.typing-cursor');
        if (cursor) cursor.remove();

        traceStatusPill.classList.remove('active');

        if (status === 'Blocked by guardrails' || isGuardrailsBlocked) {
          traceStatusPill.className = 'trace-status-pill blocked';
          traceStatusPill.textContent = 'Guardrails Blocked';
          if (stageNodes.planning) {
            stageNodes.planning.classList.remove('is-active', 'is-complete');
            stageNodes.planning.classList.add('is-blocked');
          }
          if (stageNodes.retrieving) stageNodes.retrieving.classList.add('is-skipped');
          if (stageNodes.reranking) stageNodes.reranking.classList.add('is-skipped');
          if (stageNodes.synthesizing) stageNodes.synthesizing.classList.add('is-skipped');
          return;
        }

        // Only mark stages as complete if they were actually executed (not skipped)
        Object.keys(stageNodes).forEach(key => {
          const node = stageNodes[key];
          if (node && !node.classList.contains('is-skipped') && !node.classList.contains('is-blocked')) {
            node.classList.remove('is-active');
            node.classList.add('is-complete');
          }
        });

        traceStatusPill.className = 'trace-status-pill complete';
        traceStatusPill.textContent = 'Verified Complete';
      },

      markHalted() {
        const cursor = answerContent.querySelector('.typing-cursor');
        if (cursor) cursor.remove();

        const activeNode = stageNodes[currentActiveStage];
        if (activeNode) {
          activeNode.classList.remove('is-active');
          activeNode.classList.add('is-halted');
        }

        traceStatusPill.className = 'trace-status-pill halted';
        traceStatusPill.textContent = 'Stopped';

        const notice = document.createElement('div');
        notice.className = 'notice-halted';
        notice.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line></svg>
          <span>Generation halted by user.</span>
        `;
        answerContent.appendChild(notice);
      },

      markError(msg) {
        const cursor = answerContent.querySelector('.typing-cursor');
        if (cursor) cursor.remove();

        const activeNode = stageNodes[currentActiveStage];
        if (activeNode) {
          activeNode.classList.remove('is-active');
          activeNode.classList.add('is-error');
        }

        traceStatusPill.className = 'trace-status-pill error';
        traceStatusPill.textContent = 'Error';

        const notice = document.createElement('div');
        notice.className = 'notice-error';
        notice.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          <span>${escapeHtml(msg || 'An error occurred during synthesis.')}</span>
        `;
        answerContent.appendChild(notice);
      }
    };
  }

  // ── Robust Editorial Markdown Renderer ─────────────────────────────────────
  // Architecture: Extract code blocks and inline code into shielded placeholders first
  // to prevent regex cross-corruption (e.g. array indexing [1] turning into citations)
  function renderMarkdown(rawText) {
    if (!rawText) return '';

    // Step 0: Normalize line endings to LF
    let text = rawText.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // Storage for shielded code components
    const codeBlockPlaceholders = [];
    const inlineCodePlaceholders = [];

    // Step 1: Shield Fenced Code Blocks (handles both closed blocks and streaming unclosed blocks)
    // 1a. Completed code blocks
    text = text.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const ph = `@@CODE_BLOCK_${codeBlockPlaceholders.length}@@`;
      codeBlockPlaceholders.push({ lang: lang.trim() || 'text', code: code });
      return ph;
    });

    // 1b. Streaming unclosed code block at end of text
    text = text.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*)$/g, (match, lang, code) => {
      const ph = `@@CODE_BLOCK_${codeBlockPlaceholders.length}@@`;
      codeBlockPlaceholders.push({ lang: lang.trim() || 'text', code: code, streaming: true });
      return ph;
    });

    // Step 2: Shield Inline Code (`...`)
    text = text.replace(/`([^`\n]+)`/g, (match, code) => {
      const ph = `@@INLINE_CODE_${inlineCodePlaceholders.length}@@`;
      inlineCodePlaceholders.push(code);
      return ph;
    });

    // Step 3: Escape HTML for non-code text to prevent injection
    let html = escapeHtml(text);

    // Step 4: Markdown Tables (wrapped in responsive scroll container)
    html = html.replace(/((\|[^\n]+\|\n)((?:\|[^\n]+\|\n?)+))/g, (match) => {
      const rows = match.trim().split('\n').map(r => r.trim()).filter(Boolean);
      if (rows.length < 2) return match;

      const parseCells = (row) => {
        let content = row;
        if (content.startsWith('|')) content = content.slice(1);
        if (content.endsWith('|')) content = content.slice(0, -1);
        return content.split('|').map(c => c.trim());
      };

      const headerCells = parseCells(rows[0]);
      const sep = rows[1].replace(/[\s\-\|:]/g, '');
      if (sep !== '') return match; // Not a valid markdown separator row

      let tableHtml = '<div class="table-container"><table><thead><tr>';
      headerCells.forEach(hc => {
        tableHtml += `<th>${hc}</th>`;
      });
      tableHtml += '</tr></thead><tbody>';

      for (let i = 2; i < rows.length; i++) {
        const bodyCells = parseCells(rows[i]);
        tableHtml += '<tr>';
        headerCells.forEach((_, cellIdx) => {
          tableHtml += `<td>${bodyCells[cellIdx] !== undefined ? bodyCells[cellIdx] : ''}</td>`;
        });
        tableHtml += '</tr>';
      }
      tableHtml += '</tbody></table></div>';
      return tableHtml;
    });

    // Step 5: Headers
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');

    // Step 6: Blockquotes (consecutive blockquote lines grouped together)
    html = html.replace(/(?:^&gt; (?:.*)(?:\n|$))+/gm, (match) => {
      const content = match
        .split('\n')
        .map(l => l.replace(/^&gt; ?/, '').trim())
        .filter(Boolean)
        .join('<br>');
      return `<blockquote>${content}</blockquote>`;
    });

    // Step 7: Lists (Ordered and Unordered)
    // 7a. Ordered Lists (1. 2. 3.)
    html = html.replace(/(?:^\d+\.\s+(?:.*)(?:\n|$))+/gm, (match) => {
      const items = match
        .split('\n')
        .map(l => l.replace(/^\d+\.\s+/, '').trim())
        .filter(Boolean)
        .map(i => `<li>${i}</li>`)
        .join('');
      return `<ol>${items}</ol>`;
    });

    // 7b. Unordered Lists (- or *)
    html = html.replace(/(?:^[-*]\s+(?:.*)(?:\n|$))+/gm, (match) => {
      const items = match
        .split('\n')
        .map(l => l.replace(/^[-*]\s+/, '').trim())
        .filter(Boolean)
        .map(i => `<li>${i}</li>`)
        .join('');
      return `<ul>${items}</ul>`;
    });

    // Step 8: Safe Markdown Links [title](https://...)
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="external-link">$1</a>');

    // Step 9: Bold and Italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Step 10: Citation Badges [1], [2], etc. (safe now that code is shielded!)
    html = html.replace(/\[(\d+)\]/g, (match, num) => {
      return `<button type="button" class="citation-badge" data-citation-num="${num}" title="Inspect Source [${num}]" aria-label="Inspect Source [${num}]">[${num}]</button>`;
    });

    // Step 11: Paragraphs (ignoring block containers)
    const blocks = html.split(/\n\n+/);
    html = blocks.map(b => {
      const trimmed = b.trim();
      if (!trimmed) return '';
      if (
        trimmed.startsWith('<h') || 
        trimmed.startsWith('<ul') || 
        trimmed.startsWith('<ol') || 
        trimmed.startsWith('<blockquote') || 
        trimmed.startsWith('<div') ||
        trimmed.startsWith('@@CODE_BLOCK_')
      ) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).join('\n');

    // Step 12: Restore Shielded Inline Code Placeholders
    inlineCodePlaceholders.forEach((code, idx) => {
      const escaped = escapeHtml(code);
      html = html.replace(`@@INLINE_CODE_${idx}@@`, `<code>${escaped}</code>`);
    });

    // Step 13: Restore Shielded Code Blocks
    codeBlockPlaceholders.forEach((item, idx) => {
      const escaped = escapeHtml(item.code);
      const language = escapeHtml(item.lang || 'text');
      const streamingTag = item.streaming ? ' <span class="streaming-pill">streaming</span>' : '';

      const blockHtml = `
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span>${language}${streamingTag}</span>
            <button type="button" class="code-copy-btn" data-code-index="${idx}" aria-label="Copy ${language} code">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              <span>Copy</span>
            </button>
          </div>
          <pre><code>${escaped}</code></pre>
        </div>
      `;
      html = html.replace(`@@CODE_BLOCK_${idx}@@`, blockHtml);
    });

    // Cache the original code blocks for the copy buttons
    html = html.replace(/data-code-index="(\d+)"/g, (match, idx) => {
      const block = codeBlockPlaceholders[Number(idx)];
      const safeCodeAttr = escapeHtml(block ? block.code : '');
      return `data-code="${safeCodeAttr}"`;
    });

    return html;
  }

  function bindCitationBadges(container, sources) {
    container.querySelectorAll('.citation-badge').forEach(badge => {
      badge.onclick = (e) => {
        e.preventDefault();
        const num = badge.getAttribute('data-citation-num');
        lastInspectorTrigger = badge;
        if (sources && sources.length > 0) {
          updateInspectorView(sources);
        }
        openSourceInspector(num);
      };
    });
  }

  function bindCodeCopyButtons(container) {
    container.querySelectorAll('.code-copy-btn').forEach(btn => {
      btn.onclick = () => {
        const rawCode = btn.getAttribute('data-code');
        if (rawCode) {
          const unescaped = unescapeHtml(rawCode);
          navigator.clipboard.writeText(unescaped).then(() => {
            btn.innerHTML = `
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#32765B" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
              <span style="color:#32765B;">Copied</span>
            `;
            setTimeout(() => {
              btn.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <span>Copy</span>
              `;
            }, 1800);
          });
        }
      };
    });
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function unescapeHtml(str) {
    if (!str) return '';
    const doc = new DOMParser().parseFromString(str, 'text/html');
    return doc.documentElement.textContent || str;
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // ── Launch on DOM Ready ───────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
