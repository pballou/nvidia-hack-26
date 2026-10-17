/* Acme Utilities portal: shared page behaviour (window.Acme). Vanilla JS, no deps.
   Deterministic by rule: no timers, no animation, no Date / Math.random / performance.now.

   Acme.act(type, payload = {}, opts = {}) -> Promise<{ok, toast?, error?}>
       POST /api/action. On ok: go to opts.redirect, else reload keeping the scroll position
       (the server flash then shows as the toast). Pass a redirect to drop one-shot query
       params such as ?pay=1. On error: opts.onError(error, data) if given, else an error
       toast; never reloads. While one action is in flight, further calls are ignored.
   Acme.showToast(text, kind = 'success' | 'error'), Acme.hideToast()
   Acme.openModal(el, {onClose}), Acme.closeModal(el)   el: the .modal, its backdrop, or a node inside
   Acme.closeDropdowns(), Acme.select(button)            select a chip / client-side segment
   Acme.setError(el, text)    fill (or clear with '') an .error / .alert element; toggles .field.has-error
   Acme.rootPx()              root font size in px (14 / 16 / 20): scale canvas drawing with it
   Acme.hooks                 === window.AcmeTestHooks (oracle-test hit targets; pages add keys)

   Declarative wiring (event delegation, so it works for any page markup):
     [data-act="type"] [data-payload='{json}'] [data-act-redirect="/path"]   click -> Acme.act
         On role="switch" the payload also gets [data-act-key] (default "enabled") = !aria-checked.
     button[data-dropdown="<menu id>"][aria-expanded]   toggles that in-page menu; outside click,
         Escape or picking an item closes it; ArrowUp/Down move through the items.
     [data-picker] [role=option]       picker selection (see _macros.html picker)
     [data-chips] [role=radio], .segmented > [aria-pressed]:not([data-act])   local selection
     [data-modal-open="<id>"], [data-modal-close], .modal-backdrop[data-autoshow] (opens on load)
     button[data-disclosure][aria-controls="<id>"]   toggles the panel's hidden + aria-expanded
   Every local selection fires a bubbling 'change' event on its group / picker root. */
(function () {
  'use strict';

  const hooks = (window.AcmeTestHooks = window.AcmeTestHooks || {});
  const SCROLL_KEY = 'acme.scroll';
  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
  const here = () => location.pathname + location.search;
  const shown = (el) => el.getClientRects().length > 0;
  const focus = (el) => el && el.focus({ preventScroll: true });
  let busy = false;

  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  window.addEventListener('pageshow', (e) => { if (e.persisted) busy = false; });

  // ---- toast -----------------------------------------------------------------

  function showToast(text, kind = 'success') {
    const el = document.getElementById('toast');
    if (!el) return;
    el.querySelector('.toast-text').textContent = text;
    el.classList.toggle('toast-error', kind === 'error');
    el.hidden = false;
  }

  function hideToast() {
    const el = document.getElementById('toast');
    if (el) el.hidden = true;
  }

  // ---- actions ---------------------------------------------------------------

  function reloadKeepingScroll() {
    try {
      sessionStorage.setItem(SCROLL_KEY, JSON.stringify({ path: here(), y: Math.round(window.scrollY) }));
    } catch (e) { /* storage disabled: reload at the top */ }
    location.reload();
  }

  function restoreScroll() {
    let saved = null;
    try {
      saved = JSON.parse(sessionStorage.getItem(SCROLL_KEY) || 'null');
      sessionStorage.removeItem(SCROLL_KEY);
    } catch (e) { /* ignore */ }
    if (saved && saved.path === here()) window.scrollTo(0, saved.y);
  }

  async function act(type, payload = {}, opts = {}) {
    if (busy) return { ok: false, error: 'busy' };
    busy = true;
    let data;
    try {
      const res = await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ type, payload }),
      });
      data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok !== true) {
        data = { ok: false, error: data.error || 'Something went wrong. Please try again.' };
      }
    } catch (e) {
      data = { ok: false, error: "We couldn't reach Acme Utilities. Please try again." };
    }
    if (data.ok) {
      if (opts.redirect) location.href = opts.redirect;
      else reloadKeepingScroll();
      return data; // stays busy: the page is going away
    }
    busy = false;
    if (opts.onError) opts.onError(data.error, data);
    else showToast(data.error, 'error');
    return data;
  }

  function payloadOf(el) {
    let payload = {};
    try { payload = JSON.parse(el.getAttribute('data-payload') || '{}'); } catch (e) { /* bad JSON */ }
    if (el.getAttribute('role') === 'switch') {
      payload[el.getAttribute('data-act-key') || 'enabled'] = el.getAttribute('aria-checked') !== 'true';
    }
    return payload;
  }

  // ---- dropdowns -------------------------------------------------------------

  const menuOf = (btn) => document.getElementById(btn.getAttribute('data-dropdown'));
  const menuItems = (menu) => [...menu.querySelectorAll('a[href], button:not([disabled])')].filter(shown);

  function setDropdown(btn, open) {
    const menu = menuOf(btn);
    if (!menu) return;
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    menu.hidden = !open;
  }

  function closeDropdowns(except) {
    document.querySelectorAll('[data-dropdown][aria-expanded="true"]').forEach((b) => {
      if (b !== except) setDropdown(b, false);
    });
  }

  // ---- local selection: chips, segmented, picker -----------------------------

  function select(btn) {
    const group = btn.closest('[data-chips], .segmented');
    if (!group) return;
    const attr = btn.hasAttribute('aria-pressed') ? 'aria-pressed' : 'aria-checked';
    group.querySelectorAll('[' + attr + ']').forEach((b) => b.setAttribute(attr, b === btn ? 'true' : 'false'));
    group.dataset.value = btn.dataset.value;
    group.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function pick(opt) {
    const root = opt.closest('[data-picker]');
    const btn = root.querySelector('[data-dropdown]');
    root.dataset.value = opt.dataset.value;
    root.querySelectorAll('[role="option"]').forEach((o) => o.setAttribute('aria-selected', o === opt ? 'true' : 'false'));
    root.querySelector('.picker-value').textContent = opt.textContent.trim();
    btn.classList.remove('is-placeholder');
    setDropdown(btn, false);
    focus(btn);
    root.dispatchEvent(new Event('change', { bubbles: true }));
  }

  // ---- modals ----------------------------------------------------------------

  const openModals = []; // stack of {backdrop, dialog, opener, onClose}
  const backdropOf = (el) => (el.classList.contains('modal-backdrop') ? el : el.closest('.modal-backdrop') || el);

  function openModal(el, opts = {}) {
    if (!el) return;
    const backdrop = backdropOf(el);
    if (openModals.some((m) => m.backdrop === backdrop)) return;
    const dialog = backdrop.querySelector('.modal') || backdrop;
    closeDropdowns();
    openModals.push({ backdrop, dialog, opener: document.activeElement, onClose: opts.onClose });
    backdrop.hidden = false;
    const first = dialog.querySelector('[autofocus], [data-autofocus]') ||
      [...dialog.querySelectorAll(FOCUSABLE)].find((n) => shown(n) && !n.hasAttribute('data-modal-close'));
    focus(first || dialog);
  }

  function closeModal(el) {
    if (!el) return;
    const backdrop = backdropOf(el);
    backdrop.hidden = true;
    const i = openModals.findIndex((m) => m.backdrop === backdrop);
    if (i < 0) return;
    const [m] = openModals.splice(i, 1);
    if (m.onClose) m.onClose();
    if (m.opener && m.opener !== document.body && m.opener.isConnected) focus(m.opener);
  }

  function trapTab(e) {
    const { dialog } = openModals[openModals.length - 1];
    const items = [...dialog.querySelectorAll(FOCUSABLE)].filter(shown);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    const inside = dialog.contains(document.activeElement);
    if (e.shiftKey && (!inside || document.activeElement === first)) { e.preventDefault(); focus(last); }
    else if (!e.shiftKey && (!inside || document.activeElement === last)) { e.preventDefault(); focus(first); }
  }

  // ---- inline errors ---------------------------------------------------------

  function setError(el, text) {
    if (!el) return;
    const slot = el.querySelector('[data-error-text]') || el.querySelector('span') || el;
    slot.textContent = text || '';
    el.hidden = !text;
    const field = el.closest('.field');
    if (field) field.classList.toggle('has-error', !!text);
  }

  // ---- event wiring ----------------------------------------------------------

  let pressTarget = null;
  document.addEventListener('mousedown', (e) => { pressTarget = e.target; });

  document.addEventListener('click', (e) => {
    const t = e.target instanceof Element ? e.target : e.target.parentElement;
    if (!t) return;
    const inMenu = t.closest('.dropdown-menu');
    const ddBtn = t.closest('[data-dropdown]');
    if (!inMenu && !ddBtn) closeDropdowns();

    if (ddBtn) {
      const open = ddBtn.getAttribute('aria-expanded') !== 'true';
      closeDropdowns(ddBtn);
      setDropdown(ddBtn, open);
      return;
    }
    if (t.closest('.toast-close')) { hideToast(); return; }

    const opt = t.closest('[data-picker] [role="option"]');
    if (opt) { pick(opt); return; }

    const choice = t.closest('[data-chips] [role="radio"], .segmented > [aria-pressed]:not([data-act])');
    if (choice) { select(choice); return; }

    const actEl = t.closest('[data-act]');
    if (actEl && !actEl.disabled) {
      e.preventDefault();
      if (inMenu) closeDropdowns();
      act(actEl.getAttribute('data-act'), payloadOf(actEl), { redirect: actEl.getAttribute('data-act-redirect') || undefined });
      return;
    }

    const opener = t.closest('[data-modal-open]');
    if (opener) { openModal(document.getElementById(opener.getAttribute('data-modal-open'))); return; }
    const closer = t.closest('[data-modal-close]');
    if (closer) { closeModal(closer); return; }
    if (t.classList.contains('modal-backdrop') && pressTarget === t) { closeModal(t); return; }

    const disclosure = t.closest('[data-disclosure]');
    if (disclosure) {
      const open = disclosure.getAttribute('aria-expanded') !== 'true';
      disclosure.setAttribute('aria-expanded', open ? 'true' : 'false');
      const panel = document.getElementById(disclosure.getAttribute('aria-controls'));
      if (panel) panel.hidden = !open;
      return;
    }

    if (inMenu && t.closest('a[href], button') && !t.closest('[data-keep-open]')) closeDropdowns();
  });

  document.addEventListener('keydown', (e) => {
    const t = e.target instanceof Element ? e.target : document.body;
    if (e.key === 'Escape') {
      const openBtn = document.querySelector('[data-dropdown][aria-expanded="true"]');
      if (openBtn) { e.preventDefault(); setDropdown(openBtn, false); focus(openBtn); return; }
      if (openModals.length) { e.preventDefault(); closeModal(openModals[openModals.length - 1].backdrop); }
      return;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      const btn = t.closest('[data-dropdown]');
      const menu = btn ? menuOf(btn) : t.closest('.dropdown-menu');
      if (!menu) return;
      e.preventDefault();
      if (btn) { closeDropdowns(btn); setDropdown(btn, true); }
      const items = menuItems(menu);
      if (!items.length) return;
      const step = e.key === 'ArrowDown' ? 1 : -1;
      const i = items.indexOf(document.activeElement);
      focus(items[i < 0 ? (step > 0 ? 0 : items.length - 1) : (i + step + items.length) % items.length]);
      return;
    }
    if (e.key === 'Tab' && openModals.length) trapTab(e);
  });

  // Tabbing out of an open menu closes it.
  document.addEventListener('focusin', (e) => {
    const t = e.target instanceof Element ? e.target : null;
    if (t && !t.closest('.dropdown-menu, [data-dropdown]')) closeDropdowns();
  });

  function init() {
    document.querySelectorAll('.modal-backdrop[data-autoshow]').forEach((b) => openModal(b));
    restoreScroll();
  }

  window.Acme = {
    act,
    showToast,
    hideToast,
    openModal,
    closeModal,
    closeDropdowns,
    select,
    setError,
    rootPx: () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16,
    hooks,
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
