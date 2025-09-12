const API_BASE_URL = 'http://localhost:8000'

const inputText = document.getElementsByClassName('input-text')[0]
const button = document.getElementsByClassName('sum-button')[0]
const output = document.getElementsByClassName('output')[0]

inputText.addEventListener('input', () => {
    inputText.style.height = 'auto';
    inputText.style.height = inputText.scrollHeight + 'px';
});

let isSummarizing = false

button.addEventListener('click', async () => {
    if (isSummarizing) {
        return;
    }
    isSummarizing = true;
    button.classList.add('button-disable');
    const summary = await summarize(inputText.value);
    if (summary) {
        output.textContent = summary
    }
    else {
        output.textContent = 'Có lỗi xảy ra!'
    }
    button.classList.remove('button-disable');
    isSummarizing = false;
});

async function summarize(text) {
    try {
        const response = await fetch(`${API_BASE_URL}/summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();
        return data.summary;
    }
    catch (error) {
        console.error('Error: ', error);
        return null;
    }
}