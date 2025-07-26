// Modern AI Chat Interface JavaScript
class RAGChatInterface {
    constructor() {
        this.convElement = document.getElementById('conversation');
        this.form = document.getElementById('chat-form');
        this.input = document.getElementById('prompt-input');
        this.sendButton = document.getElementById('send-button');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.errorDiv = document.getElementById('error');
        this.welcome = document.getElementById('welcome');

        this.isTyping = false;
        this.messageCount = 0;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSampleQuestions();
        this.setupTextareaAutoResize();
        this.showWelcomeMessage();
    }

    setupEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));

        // Enter key handling (Shift+Enter for new line, Enter to send)
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.form.dispatchEvent(new Event('submit'));
            }
        });

        // Input changes
        this.input.addEventListener('input', () => {
            this.updateSendButton();
            this.hideError();
        });

        // Focus management
        this.input.addEventListener('focus', () => {
            this.input.parentElement.style.borderColor = '#667eea';
        });

        this.input.addEventListener('blur', () => {
            this.input.parentElement.style.borderColor = 'var(--border-color)';
        });
    }

    setupSampleQuestions() {
        const sampleQuestions = document.querySelectorAll('.sample-question');
        sampleQuestions.forEach(question => {
            question.addEventListener('click', () => {
                const questionText = question.getAttribute('data-question');
                this.input.value = questionText;
                this.updateSendButton();
                this.input.focus();

                // Add a subtle animation
                question.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    question.style.transform = 'scale(1)';
                }, 150);
            });
        });
    }

    setupTextareaAutoResize() {
        this.input.addEventListener('input', () => {
            this.input.style.height = 'auto';
            this.input.style.height = Math.min(this.input.scrollHeight, 120) + 'px';
        });
    }

    showWelcomeMessage() {
        if (this.messageCount === 0) {
            this.welcome.style.display = 'block';
        }
    }

    hideWelcomeMessage() {
        if (this.welcome) {
            this.welcome.style.display = 'none';
        }
    }

    updateSendButton() {
        const hasText = this.input.value.trim().length > 0;
        this.sendButton.disabled = !hasText || this.isTyping;

        if (hasText && !this.isTyping) {
            this.sendButton.style.background = 'var(--primary-gradient)';
            this.sendButton.style.transform = 'scale(1)';
        } else {
            this.sendButton.style.background = 'var(--border-color)';
            this.sendButton.style.transform = 'scale(0.9)';
        }
    }

    async handleSubmit(e) {
        e.preventDefault();

        const prompt = this.input.value.trim();
        if (!prompt || this.isTyping) return;

        this.hideWelcomeMessage();
        this.addMessage('user', prompt);
        this.input.value = '';
        this.input.style.height = 'auto';
        this.updateSendButton();
        this.hideError();

        await this.sendMessage(prompt);
    }

    async sendMessage(prompt) {
        this.setTyping(true);

        try {
            console.log('Sending message:', prompt);

            const response = await fetch('/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ prompt: prompt })
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error:', errorText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);

            // Add a small delay for better UX
            await this.delay(500);

            // Check if the response has an error
            if (data.status === 'error') {
                throw new Error(data.error || 'Unknown server error');
            }

            // Add the assistant's response
            this.addMessage('assistant', data.answer || 'No response received');

            // Optionally show citations if available
            if (data.citations && Object.keys(data.citations).length > 0) {
                this.showCitations(data.citations);
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.showError(`Failed to get response: ${error.message}`);

            // Add error message as assistant response
            this.addMessage('assistant',
                '❌ Sorry, I encountered an error while processing your request. Please try again later.',
                true
            );
        } finally {
            this.setTyping(false);
        }
    }

    showCitations(citations) {
        // Add citations as a separate message or append to the last message
        let citationText = '\n\n📚 **Sources:**\n';
        for (const [source, citationList] of Object.entries(citations)) {
            citationText += `• ${source}\n`;
        }

        // You could add this as a separate message or modify the last message
        console.log('Citations:', citations);
    }

    setTyping(isTyping) {
        this.isTyping = isTyping;
        this.updateSendButton();

        if (isTyping) {
            this.typingIndicator.classList.add('active');
            this.scrollToBottom();
        } else {
            this.typingIndicator.classList.remove('active');
        }
    }

    addMessage(role, content, isError = false) {
        this.messageCount++;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        if (isError) {
            bubble.style.background = 'linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%)';
            bubble.style.color = 'white';
        }

        // Create message header
        const header = document.createElement('div');
        header.className = 'message-header';

        const avatar = document.createElement('div');
        avatar.className = 'avatar';

        const timestamp = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });

        if (role === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
            header.innerHTML = `
                <div class="avatar"><i class="fas fa-user"></i></div>
                <span>You</span>
                <span style="margin-left: auto; font-size: 0.7rem; opacity: 0.6;">${timestamp}</span>
            `;
        } else {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
            header.innerHTML = `
                <div class="avatar"><i class="fas fa-robot"></i></div>
                <span>RAG4TenMS</span>
                <span style="margin-left: auto; font-size: 0.7rem; opacity: 0.6;">${timestamp}</span>
            `;
        }

        // Create message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // Format content with basic markdown-like formatting
        const formattedContent = this.formatMessage(content);
        messageContent.innerHTML = formattedContent;

        bubble.appendChild(header);
        bubble.appendChild(messageContent);
        messageDiv.appendChild(bubble);

        // Insert before typing indicator
        this.convElement.insertBefore(messageDiv, this.typingIndicator);

        // Animate message appearance
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';

        requestAnimationFrame(() => {
            messageDiv.style.transition = 'all 0.3s ease-out';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });

        this.scrollToBottom();
    }

    formatMessage(content) {
        // Basic formatting for better readability
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 3px;">$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" style="color: #4facfe;">$1</a>');
    }

    showError(message) {
        this.errorDiv.textContent = message;
        this.errorDiv.classList.add('show');

        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        this.errorDiv.classList.remove('show');
    }

    scrollToBottom() {
        requestAnimationFrame(() => {
            this.convElement.scrollTop = this.convElement.scrollHeight;
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Enhanced features
class ChatEnhancements {
    constructor(chatInterface) {
        this.chat = chatInterface;
        this.init();
    }

    init() {
        this.setupKeyboardShortcuts();
        this.setupThemeToggle();
        this.setupCopyToClipboard();
        this.setupMessageActions();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus input
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.chat.input.focus();
            }

            // Escape to clear input
            if (e.key === 'Escape' && document.activeElement === this.chat.input) {
                this.chat.input.value = '';
                this.chat.updateSendButton();
            }
        });
    }

    setupThemeToggle() {
        // Could add theme switching functionality here
    }

    setupCopyToClipboard() {
        // Add copy functionality to messages
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-btn')) {
                const messageContent = e.target.closest('.message-bubble').querySelector('.message-content').textContent;
                navigator.clipboard.writeText(messageContent).then(() => {
                    e.target.innerHTML = '<i class="fas fa-check"></i>';
                    setTimeout(() => {
                        e.target.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 2000);
                });
            }
        });
    }

    setupMessageActions() {
        // Add hover actions to messages
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest('.message-bubble')) {
                const bubble = e.target.closest('.message-bubble');
                if (!bubble.querySelector('.message-actions')) {
                    const actions = document.createElement('div');
                    actions.className = 'message-actions';
                    actions.style.cssText = `
                        position: absolute;
                        top: -10px;
                        right: 10px;
                        background: var(--card-bg);
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        padding: 4px;
                        display: flex;
                        gap: 4px;
                        opacity: 0;
                        transition: opacity 0.2s ease;
                    `;

                    actions.innerHTML = `
                        <button class="copy-btn" style="background: none; border: none; color: var(--text-secondary); cursor: pointer; padding: 4px; border-radius: 3px;">
                            <i class="fas fa-copy" style="font-size: 12px;"></i>
                        </button>
                    `;

                    bubble.style.position = 'relative';
                    bubble.appendChild(actions);

                    setTimeout(() => {
                        actions.style.opacity = '1';
                    }, 100);
                }
            }
        });

        document.addEventListener('mouseleave', (e) => {
            if (e.target.closest('.message')) {
                const actions = e.target.closest('.message').querySelector('.message-actions');
                if (actions) {
                    actions.style.opacity = '0';
                    setTimeout(() => {
                        if (actions.parentNode) {
                            actions.parentNode.removeChild(actions);
                        }
                    }, 200);
                }
            }
        });
    }
}

// Initialize the chat interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatInterface = new RAGChatInterface();
    const enhancements = new ChatEnhancements(chatInterface);

    // Add some visual feedback for loading
    document.body.classList.add('fade-in');

    console.log('🤖 Multilingual RAG Chat Interface initialized');
    console.log('💡 Tip: Use Ctrl+K to focus the input field');
});

// Export for potential external use
window.RAGChatInterface = RAGChatInterface;
