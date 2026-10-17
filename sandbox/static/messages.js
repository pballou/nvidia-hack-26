/* Messages > New message: inline validation, then Acme.act('messages.send').
   The inbox and message view need no page JS: stars and mark-all-read use [data-act]. */
(function () {
  'use strict';

  const form = document.getElementById('compose-form');
  if (!form) return;
  const picker = form.querySelector('[data-picker]');
  const toButton = picker.querySelector('[data-dropdown]');
  const subject = document.getElementById('compose-subject-input');
  const body = document.getElementById('compose-body-input');
  const errorEl = form.querySelector('[data-testid="compose-error"]');

  // Returns the error text ('' when the form is valid) and marks the fields that need fixing.
  function problem() {
    const noTo = !picker.dataset.value;
    const noBody = !body.value.trim();
    picker.closest('.field').classList.toggle('has-error', noTo);
    body.closest('.field').classList.toggle('has-error', noBody);
    if (noTo && noBody) return 'Choose a recipient and write your message.';
    if (noTo) return 'Choose a recipient.';
    if (noBody) return 'Write a message before sending.';
    return '';
  }

  function validate() {
    const text = problem();
    Acme.setError(errorEl, text);
    return !text;
  }

  // Once an error shows, re-check as the user fixes it.
  const recheck = () => { if (!errorEl.hidden) validate(); };
  picker.addEventListener('change', recheck);
  body.addEventListener('input', recheck);

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!validate()) {
      (picker.dataset.value ? body : toButton).focus();
      return;
    }
    Acme.act('messages.send', { to: picker.dataset.value, subject: subject.value, body: body.value }, {
      redirect: '/messages?tab=sent',
      onError: (err) => Acme.setError(errorEl, err),
    });
  });
})();
