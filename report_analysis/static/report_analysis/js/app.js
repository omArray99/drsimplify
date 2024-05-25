document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const chatForm = document.getElementById('chatForm');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        fetch('/upload-explain/', {
            method: 'POST',
            body: formData
        }).then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'Medical_Report_Explanation.pdf';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        }).catch(error => {
            console.error('Error:', error);
        });
    });

    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const questionText = document.getElementById('questionInput').value;
        fetch('/ask-question/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: questionText})
        }).then(response => response.json())
        .then(data => {
            document.getElementById('answerText').innerText = data.answer || "Failed to get an answer.";
            document.getElementById('answerOutput').classList.remove('hidden');
        }).catch(error => {
            console.error('Error:', error);
        });
        document.getElementById('questionInput').value = ''; // Clear the input after sending
    });
});
