/* Settings page: profile form validation and save. The notification switches and the
   accessibility controls are wired declaratively by app.js ([data-act]). */
(function () {
  'use strict';

  const form = document.getElementById('profile-form');
  if (!form) return;

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const FIELDS = {
    name: ['profile-name', (v) => (v ? '' : 'Enter your full name')],
    email: ['profile-email', (v) => (!v ? 'Enter your email address'
      : EMAIL_RE.test(v) ? '' : 'Enter a valid email address')],
    phone: ['profile-phone', (v) => {
      const digits = v.replace(/\D/g, '');
      if (!v) return 'Enter your phone number';
      return digits.length === 10 || (digits.length === 11 && digits[0] === '1') ? '' : 'Enter a 10-digit phone number';
    }],
  };
  const formError = document.getElementById('profile-form-error');

  function setFieldError(id, msg) {
    Acme.setError(document.getElementById(id + '-error'), msg);
    document.getElementById(id).setAttribute('aria-invalid', msg ? 'true' : 'false');
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const values = {};
    let firstBad = null;
    for (const [key, [id, check]] of Object.entries(FIELDS)) {
      values[key] = document.getElementById(id).value.trim();
      const msg = check(values[key]);
      setFieldError(id, msg);
      if (msg && !firstBad) firstBad = document.getElementById(id);
    }
    Acme.setError(formError, '');
    if (firstBad) { firstBad.focus(); return; }
    Acme.act('settings.profile', values, { onError: (msg) => Acme.setError(formError, msg) });
  });

  // Editing a field clears its error; "Discard changes" clears them all.
  form.addEventListener('input', (e) => {
    const id = e.target.id;
    if (Object.values(FIELDS).some(([fid]) => fid === id)) setFieldError(id, '');
  });
  form.addEventListener('reset', () => {
    Object.values(FIELDS).forEach(([id]) => setFieldError(id, ''));
    Acme.setError(formError, '');
  });
})();
