export function createConsole(els) {
  const commands = new Map();
  let drag = null;
  let completionSession = null;

  function log(message, type = 'system') {
    const row = document.createElement('div');
    row.className = `console-line ${type}`;
    row.innerHTML = `<span class="console-time">[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}]</span><span class="console-type">${type.toUpperCase()}</span><span class="console-message"></span>`;
    row.querySelector('.console-message').textContent = message;
    els.consoleOutput.appendChild(row);
    els.consoleOutput.scrollTop = els.consoleOutput.scrollHeight;
  }

  function toggle(force) {
    const show = typeof force === 'boolean' ? force : els.consolePanel.classList.contains('hidden');
    els.consolePanel.classList.toggle('hidden', !show);
    if (show && els.consoleInput) {
      requestAnimationFrame(() => {
        els.consoleInput.focus();
        const end = els.consoleInput.value.length;
        els.consoleInput.setSelectionRange(end, end);
      });
    }
  }

  function register(name, run, helpOrOptions) {
    const options = typeof helpOrOptions === 'object' && helpOrOptions !== null
      ? helpOrOptions
      : { help: helpOrOptions };
    commands.set(name, { run, help: options.help, complete: options.complete });
  }

  async function execute(input, options = {}) {
    const text = String(input || '').trim();
    if (!text) return;
    const [name, ...args] = text.split(/\s+/);
    const command = commands.get(name);
    if (options.echo !== false) log(`> ${text}`, 'command');
    if (!command) {
      log(`Unknown command: ${name}`, 'error');
      toggle(true);
      return;
    }
    try {
      await command.run(args, text);
    } catch (error) {
      log(String(error), 'error');
      toggle(true);
    }
  }

  function uniqueSorted(values) {
    return [...new Set(values.filter(Boolean).map(String))].sort((a, b) => a.localeCompare(b));
  }

  function commonPrefix(values) {
    if (!values.length) return '';
    return values.reduce((prefix, value) => {
      let index = 0;
      while (index < prefix.length && index < value.length && prefix[index] === value[index]) index += 1;
      return prefix.slice(0, index);
    });
  }

  function tokenAt(value, cursor) {
    const left = value.slice(0, cursor);
    const right = value.slice(cursor);
    const leftMatch = left.match(/\S*$/);
    const rightMatch = right.match(/^\S*/);
    const start = cursor - (leftMatch ? leftMatch[0].length : 0);
    const end = cursor + (rightMatch ? rightMatch[0].length : 0);
    return { start, end, value: value.slice(start, end) };
  }

  function getCompletionOptions(value, cursor) {
    const token = tokenAt(value, cursor);
    const beforeToken = value.slice(0, token.start).trimStart();
    const partsBefore = beforeToken ? beforeToken.trimEnd().split(/\s+/) : [];
    const commandName = partsBefore[0] || (token.start > 0 ? value.trimStart().split(/\s+/)[0] : '');
    const command = commands.get(commandName);
    if (!command || partsBefore.length === 0) {
      return uniqueSorted([...commands.keys()]).filter((name) => name.startsWith(token.value));
    }
    if (typeof command.complete !== 'function') return [];
    const argsBefore = partsBefore.slice(1);
    return uniqueSorted(command.complete({
      argsBefore,
      commandName,
      cursor,
      token: token.value,
      value
    })).filter((item) => item.startsWith(token.value));
  }

  function replaceToken(input, token, replacement, addSpace) {
    const before = input.value.slice(0, token.start);
    const after = input.value.slice(token.end);
    const nextValue = `${before}${replacement}${addSpace ? ' ' : ''}${after}`;
    const nextCursor = before.length + replacement.length + (addSpace ? 1 : 0);
    input.value = nextValue;
    input.setSelectionRange(nextCursor, nextCursor);
  }

  function completeInput(event) {
    const input = els.consoleInput;
    const value = input.value;
    const cursor = input.selectionStart ?? value.length;
    const token = tokenAt(value, cursor);
    const matches = getCompletionOptions(value, cursor);
    if (!matches.length) {
      completionSession = null;
      return;
    }

    const sessionKey = `${value.slice(0, token.start)}|${token.value}|${value.slice(token.end)}`;
    const prefix = commonPrefix(matches);
    if (prefix.length > token.value.length) {
      replaceToken(input, token, prefix, matches.length === 1);
      completionSession = null;
      return;
    }

    const sameSession = completionSession
      && completionSession.key === sessionKey
      && completionSession.matches.join('\n') === matches.join('\n');
    const index = sameSession ? (completionSession.index + 1) % matches.length : 0;
    completionSession = { key: sessionKey, matches, index };
    replaceToken(input, token, matches[index], true);
    if (!sameSession && matches.length > 1) log(matches.join('    '), 'system');
  }

  function attachDrag() {
    els.consoleHead.addEventListener('pointerdown', (event) => {
      if (event.target === els.consoleClose) return;
      const rect = els.consolePanel.getBoundingClientRect();
      drag = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      els.consoleHead.setPointerCapture(event.pointerId);
    });
    els.consoleHead.addEventListener('pointermove', (event) => {
      if (!drag) return;
      const left = Math.max(8, Math.min(window.innerWidth - els.consolePanel.offsetWidth - 8, event.clientX - drag.x));
      const top = Math.max(8, Math.min(window.innerHeight - els.consolePanel.offsetHeight - 8, event.clientY - drag.y));
      els.consolePanel.style.left = `${left}px`;
      els.consolePanel.style.top = `${top}px`;
      els.consolePanel.style.right = 'auto';
      els.consolePanel.style.bottom = 'auto';
      els.consolePanel.style.transform = 'none';
    });
    els.consoleHead.addEventListener('pointerup', () => {
      drag = null;
    });
  }

  function attachInput() {
    if (!els.consoleForm || !els.consoleInput) return;
    els.consoleInput.addEventListener('keydown', (event) => {
      if (event.key === 'Tab') {
        event.preventDefault();
        completeInput(event);
        return;
      }
      completionSession = null;
    });
    els.consoleForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const command = els.consoleInput.value;
      els.consoleInput.value = '';
      execute(command);
    });
  }

  attachDrag();
  attachInput();
  return { log, toggle, register, execute, commands };
}
