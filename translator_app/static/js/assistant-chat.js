document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('assistant-chat-toggle');
    const panel = document.getElementById('assistant-chat-panel');
    const closeButton = document.getElementById('assistant-chat-close');
    const messages = document.getElementById('assistant-chat-messages');
    const form = document.getElementById('assistant-chat-form');
    const input = document.getElementById('assistant-chat-input');
    const sendButton = document.getElementById('assistant-chat-send');
    const status = document.getElementById('assistant-chat-status');
    const suggestions = document.querySelector('.assistant-chat-suggestions');
    const modelProfileSelect = document.getElementById('model-profile-select');

    if (!toggle || !panel || !messages || !form || !input || !sendButton) return;

    function getCsrfToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
    }

    function appendMessage(text, className) {
        const message = document.createElement('div');
        message.className = `assistant-chat-message ${className}`;
        message.textContent = text.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*(.*?)\*/g, '$1');
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }

    async function sendQuestion(question) {
        const query = question.trim();
        if (!query) return;

        appendMessage(query, 'assistant-user-message');
        input.value = '';
        input.disabled = true;
        sendButton.disabled = true;
        if (suggestions) suggestions.hidden = true;
        if (status) status.textContent = 'Checking the site help…';

        try {
            const response = await fetch('/api/ai_chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({
                    query,
                    model_profile: modelProfileSelect?.value || 'wlasl100'
                })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `Assistant request failed (${response.status}).`);
            }
            if (data.status !== 'success' || typeof data.response !== 'string') {
                throw new Error('The assistant returned an invalid response.');
            }
            appendMessage(data.response, 'assistant-reply');
        } catch (error) {
            console.error('Assistant chat error:', error);
            appendMessage(
                error.message || 'The assistant could not reply. Please try again.',
                'assistant-reply assistant-error-message'
            );
        } finally {
            input.disabled = false;
            sendButton.disabled = false;
            if (status) status.textContent = '';
            if (!panel.hidden) input.focus();
        }
    }

    toggle.addEventListener('click', () => {
        panel.hidden = false;
        toggle.setAttribute('aria-expanded', 'true');
        input.focus();
    });

    closeButton?.addEventListener('click', () => {
        panel.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
        toggle.focus();
    });

    form.addEventListener('submit', event => {
        event.preventDefault();
        sendQuestion(input.value);
    });

    suggestions?.addEventListener('click', event => {
        const suggestion = event.target.closest('[data-assistant-question]');
        if (!suggestion) return;
        sendQuestion(suggestion.dataset.assistantQuestion);
    });

    window.addEventListener('keydown', event => {
        if (event.key !== 'Escape' || panel.hidden) return;
        panel.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
        toggle.focus();
    });
});
