/* Help center: the call-back request form. The FAQ accordions use the shared
   [data-disclosure] wiring in app.js. */
(function () {
  'use strict';

  const form = document.getElementById('callback-form');
  if (!form) return;
  const topics = document.getElementById('callback-topic');
  const message = document.getElementById('callback-message');
  const topicError = document.getElementById('callback-topic-error');
  const messageError = document.getElementById('callback-message-error');

  function setError(input, el, text) {
    Acme.setError(el, text);
    if (text) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }

  topics.addEventListener('change', () => setError(topics, topicError, ''));
  message.addEventListener('input', () => {
    if (message.value.trim()) setError(message, messageError, '');
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const topic = topics.dataset.value;
    const text = message.value.trim();
    setError(topics, topicError, topic ? '' : 'Choose what the call is about.');
    setError(message, messageError, text ? '' : 'Tell us what to call about.');
    if (!topic) { topics.querySelector('[role="radio"]').focus(); return; }
    if (!text) { message.focus(); return; }
    Acme.act('help.contact', { topic, message: text });
  });
})();
