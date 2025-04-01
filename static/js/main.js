document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const filesList = document.getElementById('filesList');
    const extractedText = document.getElementById('extractedText');
    const previewImage = document.getElementById('previewImage');

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // File input handler
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        Array.from(files).forEach(file => {
            if (file.type === 'application/pdf' || file.type.startsWith('image/')) {
                addFileToList(file);
                processFile(file);
            } else {
                alert('Please upload only PDF or image files.');
            }
        });
    }

    function addFileToList(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const icon = document.createElement('img');
        icon.src = file.type === 'application/pdf' 
            ? '/static/images/pdf-icon.svg' 
            : '/static/images/image-icon.svg';
        icon.alt = file.type === 'application/pdf' ? 'PDF' : 'Image';
        
        const fileName = document.createElement('span');
        fileName.textContent = file.name;
        
        const progress = document.createElement('div');
        progress.className = 'progress';
        progress.style.width = '0%';
        
        fileItem.appendChild(icon);
        fileItem.appendChild(fileName);
        fileItem.appendChild(progress);
        filesList.appendChild(fileItem);
        
        return progress;
    }

    async function processFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Processing failed');
            }

            const result = await response.json();
            
            // Update the extracted text
            extractedText.value = result.text;
            
            // Update the preview image if it's an image file
            if (file.type.startsWith('image/')) {
                previewImage.src = URL.createObjectURL(file);
                previewImage.style.display = 'block';
            } else {
                previewImage.style.display = 'none';
            }
            
            // Update progress
            const progress = filesList.lastElementChild.querySelector('.progress');
            progress.style.width = '100%';
            progress.style.backgroundColor = '#27ae60';
            
        } catch (error) {
            console.error('Error processing file:', error);
            alert('Error processing file. Please try again.');
            
            // Update progress to show error
            const progress = filesList.lastElementChild.querySelector('.progress');
            progress.style.width = '100%';
            progress.style.backgroundColor = '#e74c3c';
        }
    }
}); 