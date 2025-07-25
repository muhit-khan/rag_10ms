const convElement = document.getElementById('conversation');
const form = document.getElementById('chat-form');
const input = document.getElementById('prompt-input');
const spinner = document.getElementById('spinner');
const errorDiv = document.getElementById('error');

function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = role;
    div.innerHTML = `<b>${role === 'user' ? 'You' : 'RAG'}:</b> ${content}`;
    convElement.appendChild(div);
    convElement.scrollTop = convElement.scrollHeight;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = input.value.trim();
    if (!prompt) return;
    addMessage('user', prompt);
    input.value = '';
    spinner.classList.add('active');
    errorDiv.classList.add('d-none');
    try {
        const res = await fetch('/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        if (!res.ok) throw new Error('Network error');
        const data = await res.json();
        addMessage('assistant', data.answer);
    } catch (err) {
        errorDiv.classList.remove('d-none');
        console.error(err);
    } finally {
        spinner.classList.remove('active');
    }
});
