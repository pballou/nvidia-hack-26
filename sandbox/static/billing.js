/* Billing page: the 2-step pay flow (amount -> confirm -> billing.pay).
   Statement downloads and the Autopay switch are declarative [data-act] (app.js). */
(function () {
  'use strict';

  const byTestId = (id) => document.querySelector('[data-testid="' + id + '"]');
  // ?pay=1 (the dashboard's Pay bill) opens the dialog once, server-side. Drop it from the
  // URL right away so later reloads (autopay, downloads, any action) don't reopen it.
  const here = new URL(location.href);
  if (here.searchParams.has('pay')) {
    here.searchParams.delete('pay');
    history.replaceState(history.state, '', here.pathname + here.search + here.hash);
  }

  const payDialog = byTestId('pay-dialog');
  if (!payDialog) return; // nothing due: no pay flow

  const data = JSON.parse(document.getElementById('billing-data').textContent);
  const confirmDialog = byTestId('confirm-dialog');
  const form = document.getElementById('pay-form');
  const amountInput = byTestId('pay-amount');
  const amountGroup = amountInput.closest('.input-group');
  const errorEl = byTestId('pay-error');
  const otherOption = form.querySelector('.bill-option-other');
  const radios = [...form.querySelectorAll('input[name="pay-choice"]')];
  const confirmBtn = byTestId('pay-confirm');
  let pendingCents = 0;

  // Integer cents -> "$1,234.56" (no float math, no locale).
  function money(cents) {
    const dollars = String(Math.floor(cents / 100)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return '$' + dollars + '.' + String(cents % 100).padStart(2, '0');
  }

  // "50", "50.5", "$50.00", " $1,000 ", ".75" -> {cents} or {error}. Pure string parsing.
  function parseCents(raw) {
    const s = raw.trim().replace(/^\$\s*/, '').replace(/,/g, '');
    if (!s) return { error: 'Enter the amount you want to pay' };
    const m = /^(\d*)(?:\.(\d*))?$/.exec(s);
    if (!m || (!m[1] && !m[2])) return { error: 'Enter a dollar amount, like 50 or 50.25' };
    const frac = m[2] || '';
    if (frac.length > 2) return { error: 'Use no more than 2 digits after the decimal point' };
    const whole = m[1].replace(/^0+(?=\d)/, '');
    if (whole.length > 9) return { cents: Infinity };
    return { cents: Number(whole || '0') * 100 + Number((frac + '00').slice(0, 2)) };
  }

  const choice = () => (radios.find((r) => r.checked) || radios[0]).value;

  function setChoice(value) {
    radios.forEach((r) => {
      r.checked = r.value === value;
      r.closest('.bill-option').classList.toggle('is-selected', r.checked);
    });
  }

  function showError(text) {
    Acme.setError(errorEl, text);
    amountGroup.classList.toggle('has-error', !!text && choice() === 'other');
  }

  function validate() {
    if (choice() === 'full') return { cents: data.balance_cents };
    const r = parseCents(amountInput.value);
    if (r.error) return r;
    if (r.cents <= 0) return { error: 'Enter an amount greater than $0.00' };
    if (r.cents > data.balance_cents) return { error: 'You can pay at most ' + money(data.balance_cents) };
    return r;
  }

  function resetForm() {
    form.reset();
    setChoice('full');
    showError('');
  }

  function backToAmount(error) {
    Acme.closeModal(confirmDialog);
    Acme.openModal(payDialog);
    if (choice() === 'other') amountInput.focus({ preventScroll: true });
    if (error) showError(error);
  }

  // Focusing or typing in the amount field selects "Other amount"; so does clicking anywhere on that option.
  amountInput.addEventListener('focus', () => setChoice('other'));
  amountInput.addEventListener('input', () => { setChoice('other'); showError(''); });
  otherOption.addEventListener('click', (e) => {
    if (e.target === amountInput) return;
    setChoice('other');
    amountInput.focus({ preventScroll: true });
  });
  radios.forEach((r) => r.addEventListener('change', () => { setChoice(choice()); showError(''); }));

  byTestId('pay-open').addEventListener('click', () => {
    resetForm();
    Acme.openModal(payDialog);
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const r = validate();
    if (r.error) {
      showError(r.error);
      if (choice() === 'other') amountInput.focus({ preventScroll: true });
      return;
    }
    pendingCents = r.cents;
    byTestId('confirm-text').textContent = 'Pay ' + money(pendingCents) + ' from ' + data.method + '?';
    document.getElementById('confirm-amount').textContent = money(pendingCents);
    document.getElementById('confirm-remaining').textContent = money(data.balance_cents - pendingCents);
    Acme.closeModal(payDialog);
    Acme.openModal(confirmDialog);
  });

  byTestId('pay-back').addEventListener('click', () => backToAmount());

  confirmBtn.addEventListener('click', () => {
    confirmBtn.disabled = true;
    Acme.act('billing.pay', { amount_cents: pendingCents }, {
      onError: (error) => { confirmBtn.disabled = false; backToAmount(error); },
    });
  });

  resetForm(); // ignore any form state the browser restored on reload
})();
