/**
 * Enterprise Agentic Assistant — Modern Client Controller
 * Features: SSE Stream Processing, AbortController (Stop Response),
 * Markdown Rendering, Code Highlighting with Copy, Thought Process Accordions,
 * Source Context Visualizer, and Session Memory Management.
 */

(function () {
  'use strict';

  // ── State Management ────────────────────────────────────────────────────────
  let currentSessionId = getOrCreateSessionId();
  let currentAbortController = null;
  let isGenerating = false;

  // ── DOM References ─────────────────────────────────────────────────────────
  const sidebar = document.getElementById('sidebar');
  const mobileToggleBtn = document.getElementById('mobileToggleBtn');
  const mobileCloseBtn = document.getElementById('mobileCloseBtn');
  const memoryIdDisplay = document.getElementById('memoryIdDisplay');
  const copyMemoryIdBtn = document.getElementById('copyMemoryIdBtn');
  const clearMemoryBtn = document.getElementById('clearMemoryBtn');
  const messagesContainer = document.getElementById('messagesContainer');
  const heroWelcome = document.getElementById('heroWelcome');
  const chatThread = document.getElementById('chatThread');
  const promptForm = document.getElementById('promptForm');
  const promptInput = document.getElementById('promptInput');
  const sendBtn = document.getElementById('sendBtn');
  const generationBar = document.getElementById('generationBar');
  const genStatusText = document.getElementById('genStatusText');
  const stopResponseBtn = document.getElementById('stopResponseBtn');
  const toast = document.getElementById('toast');

  // ── Initialization ─────────────────────────────────────────────────────────
  function init() {
    updateMemoryDisplay();
    setupEventListeners();
    setupTextareaAutoResize();
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
    memoryIdDisplay.textContent = currentSessionId.substring(0, 8);
  }

  // ── Event Listeners ────────────────────────────────────────────────────────
  function setupEventListeners() {
    // Mobile Sidebar Toggles
    if (mobileToggleBtn) {
      mobileToggleBtn.addEventListener('click', () => sidebar.classList.add('open'));
    }
    if (mobileCloseBtn) {
      mobileCloseBtn.addEventListener('click', () => sidebar.classList.remove('open'));
    }

    // Copy Memory ID
    copyMemoryIdBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(currentSessionId).then(() => {
        showToast('Memory ID copied to clipboard');
      });
    });

    // Clear Memory Button
    clearMemoryBtn.addEventListener('click', handleClearMemory);

    // Stop Response Button
    stopResponseBtn.addEventListener('click', handleStopResponse);

    // Prompt Form Submit
    promptForm.addEventListener('submit', (e) => {
      e.preventDefault();
      handleSubmit();
    });

    // Enter to Send, Shift+Enter for Newline
    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    });

    // Suggestions & Hero Chips click-to-send
    document.querySelectorAll('.suggestion-item, .hero-chip').forEach(el => {
      el.addEventListener('click', () => {
        const prompt = el.getAttribute('data-prompt');
        if (prompt && !isGenerating) {
          promptInput.value = prompt;
          handleSubmit();
        }
      });
    });
  }

  function setupTextareaAutoResize() {
    promptInput.addEventListener('input', () => {
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + 'px';
      sendBtn.disabled = promptInput.value.trim() === '';
    });
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2600);
  }

  // ── Memory Clear Handler ───────────────────────────────────────────────────
  async function handleClearMemory() {
    if (isGenerating) {
      handleStopResponse();
    }

    try {
      await fetch('/memory/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: currentSessionId })
      });
    } catch (e) {
      console.warn('Memory reset signal sent:', e);
    }

    // Rotate to a fresh session ID
    currentSessionId = (typeof crypto.randomUUID === 'function') 
      ? crypto.randomUUID() 
      : 'sess_' + Math.random().toString(36).substring(2, 12);
    localStorage.setItem('rag_session_id', currentSessionId);
    updateMemoryDisplay();

    // Clear UI thread
    chatThread.innerHTML = '';
    heroWelcome.style.display = 'block';
    showToast('Conversation history and agent memory cleared');
  }

  // ── Stop Response Handler ──────────────────────────────────────────────────
  function handleStopResponse() {
    if (currentAbortController && isGenerating) {
      currentAbortController.abort();
      currentAbortController = null;
      setGenerationActive(false);
      showToast('Response stopped by user');
      
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

    // Hide welcome hero on first message
    if (heroWelcome.style.display !== 'none') {
      heroWelcome.style.display = 'none';
    }

    // Render User Message
    appendUserMessage(query);

    // Prepare Assistant Message Container
    const assistantCard = createAssistantCard();
    chatThread.appendChild(assistantCard.container);
    scrollToBottom();

    // Setup AbortController for cancellation
    currentAbortController = new AbortController();
    setGenerationActive(true, 'Evaluating intent & guardrails...');

    let accumulatedMarkdown = '';
    const thoughts = [];
    const sourcesList = [];

    try {
      const response = await fetch('/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: query, thread_id: currentSessionId }),
        signal: currentAbortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // keep partial line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonStr = trimmed.replace(/^data:\s*/, '');
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'thought') {
              thoughts.push(data.step);
              assistantCard.addThought(data.step);
              genStatusText.textContent = data.step;
              scrollToBottom();
            } else if (data.type === 'sources') {
              if (Array.isArray(data.sources)) {
                sourcesList.push(...data.sources);
                assistantCard.renderSources(sourcesList);
              }
            } else if (data.type === 'token') {
              accumulatedMarkdown += data.content;
              assistantCard.updateAnswer(accumulatedMarkdown);
              scrollToBottom();
            } else if (data.type === 'done') {
              assistantCard.finalize(thoughts.length);
              setGenerationActive(false);
            }
          } catch (jsonErr) {
            console.warn('Failed to parse SSE payload:', jsonStr, jsonErr);
          }
        }
      }

      assistantCard.finalize(thoughts.length);
    } catch (err) {
      if (err.name === 'AbortError') {
        assistantCard.appendHaltedNotice();
      } else {
        console.error('Streaming request failed:', err);
        assistantCard.appendErrorNotice(err.message);
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

    // Header
    const header = document.createElement('div');
    header.className = 'assistant-header';
    header.innerHTML = `
      <div class="assistant-avatar">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
      </div>
      <span>Enterprise Assistant</span>
    `;
    container.appendChild(header);

    // Thought Process Box
    const thoughtBox = document.createElement('div');
    thoughtBox.className = 'thought-box open';
    thoughtBox.innerHTML = `
      <div class="thought-header">
        <div class="thought-title-group">
          <span class="thought-icon-spinner">⚙️</span>
          <span class="thought-summary-title">Agent Reasoning...</span>
        </div>
        <svg class="thought-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </div>
      <div class="thought-steps"></div>
    `;
    const thoughtSteps = thoughtBox.querySelector('.thought-steps');
    const thoughtSummaryTitle = thoughtBox.querySelector('.thought-summary-title');
    const thoughtHeader = thoughtBox.querySelector('.thought-header');

    thoughtHeader.addEventListener('click', () => {
      thoughtBox.classList.toggle('open');
    });

    container.appendChild(thoughtBox);

    // Answer Content
    const answerContent = document.createElement('div');
    answerContent.className = 'answer-content';
    answerContent.innerHTML = '<span class="typing-cursor">▌</span>';
    container.appendChild(answerContent);

    // Sources Accordion Container
    const sourcesBox = document.createElement('div');
    sourcesBox.className = 'sources-accordion';
    sourcesBox.style.display = 'none';
    sourcesBox.innerHTML = `
      <div class="sources-header">
        <span>📄 Retrieved Context (<strong class="sources-count">0</strong> sources)</span>
        <svg class="sources-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </div>
      <div class="sources-list"></div>
    `;
    const sourcesHeader = sourcesBox.querySelector('.sources-header');
    const sourcesListEl = sourcesBox.querySelector('.sources-list');
    const sourcesCountEl = sourcesBox.querySelector('.sources-count');

    sourcesHeader.addEventListener('click', () => {
      sourcesBox.classList.toggle('open');
    });
    container.appendChild(sourcesBox);

    return {
      container,
      addThought(step) {
        const item = document.createElement('div');
        item.className = 'thought-step-item';
        item.innerHTML = `<span class="step-prefix">↳</span><span>${escapeHtml(step)}</span>`;
        thoughtSteps.appendChild(item);
      },
      updateAnswer(mdText) {
        answerContent.innerHTML = renderMarkdown(mdText) + '<span class="typing-cursor">▌</span>';
        bindCodeCopyButtons(answerContent);
      },
      renderSources(sources) {
        if (!sources || sources.length === 0) return;
        sourcesBox.style.display = 'block';
        sourcesCountEl.textContent = sources.length;
        sourcesListEl.innerHTML = '';

        sources.forEach((s, idx) => {
          const card = document.createElement('div');
          card.className = 'source-card';
          card.innerHTML = `
            <div class="source-card-header">
              <span class="source-badge">Chunk ${idx + 1} · ${escapeHtml(s.source || 'Document')}</span>
            </div>
            <div class="source-preview">${escapeHtml(s.content || '')}</div>
          `;
          sourcesListEl.appendChild(card);
        });
      },
      finalize(stepsCount) {
        const spinner = thoughtBox.querySelector('.thought-icon-spinner');
        if (spinner) spinner.remove();
        thoughtSummaryTitle.textContent = `${stepsCount} reasoning step${stepsCount === 1 ? '' : 's'} completed`;
        thoughtBox.classList.remove('open'); // auto-collapse when generation finishes

        const cursor = answerContent.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
      },
      appendHaltedNotice() {
        const cursor = answerContent.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
        const notice = document.createElement('div');
        notice.style.cssText = 'margin-top: 10px; font-size: 12px; color: #ef4444; font-style: italic;';
        notice.textContent = '⏹️ Generation stopped by user.';
        answerContent.appendChild(notice);
      },
      appendErrorNotice(msg) {
        const cursor = answerContent.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
        const err = document.createElement('div');
        err.style.cssText = 'margin-top: 10px; font-size: 12px; color: #ef4444;';
        err.textContent = `⚠️ Error: ${msg}`;
        answerContent.appendChild(err);
      }
    };
  }

  // ── Lightweight Markdown Renderer ──────────────────────────────────────────
  function renderMarkdown(text) {
    if (!text) return '';

    let html = text;

    // Fenced Code blocks
    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'text';
      const cleanCode = escapeHtml(code.trim());
      return `
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span>${language}</span>
            <button class="code-copy-btn" data-code="${cleanCode}">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              Copy
            </button>
          </div>
          <pre><code>${cleanCode}</code></pre>
        </div>
      `;
    });

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold and Italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Inline Code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Bulleted Lists
    html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>[\s\S]*?<\/li>)/gm, '<ul>$1</ul>');

    // Clean duplicate <ul> wraps
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Paragraphs (double line breaks)
    const paragraphs = html.split(/\n\n+/);
    html = paragraphs.map(p => {
      const trimmed = p.trim();
      if (trimmed.startsWith('<h') || trimmed.startsWith('<ul') || 
          trimmed.startsWith('<ol') || trimmed.startsWith('<div') || 
          trimmed.startsWith('<blockquote') || trimmed.startsWith('<table')) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
  }

  function bindCodeCopyButtons(container) {
    container.querySelectorAll('.code-copy-btn').forEach(btn => {
      btn.onclick = () => {
        const rawCode = btn.getAttribute('data-code');
        if (rawCode) {
          navigator.clipboard.writeText(rawCode).then(() => {
            btn.innerHTML = `
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
              Copied!
            `;
            setTimeout(() => {
              btn.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                Copy
              `;
            }, 2000);
          });
        }
      };
    });
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // ── Launch ─────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);

})();
