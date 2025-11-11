document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const imageUpload = document.getElementById('imageUpload');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const previewImage = document.getElementById('previewImage');
    const resultStatus = document.getElementById('resultStatus');
    const resultCategories = document.getElementById('resultCategories');
    
    let selectedFile = null;
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            window.scrollTo({
                top: targetSection.offsetTop - 80,
                behavior: 'smooth'
            });
            
            // Update active link
            document.querySelectorAll('nav a').forEach(navLink => {
                navLink.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    // Handle file upload via click
    uploadArea.addEventListener('click', function() {
        imageUpload.click();
    });
    
    // Handle file upload via drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.backgroundColor = 'rgba(74, 111, 220, 0.1)';
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.style.backgroundColor = '';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.backgroundColor = '';
        
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    // Handle file selection
    imageUpload.addEventListener('change', function() {
        if (this.files.length) {
            handleFileSelect(this.files[0]);
        }
    });
    
    // Process selected file
    function handleFileSelect(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showError('Please select an image file (JPEG, PNG, etc.)');
            return;
        }
        
        selectedFile = file;
        
        // Display preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            resultStatus.textContent = 'Image selected. Click "Analyze Scan" to process.';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    // Handle analyze button click
    analyzeBtn.addEventListener('click', function() {
        if (!selectedFile) {
            showError('Please select an image first');
            return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        resultStatus.textContent = 'Analyzing brain scan...';
        
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Send to API
        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data.results);
        })
        .catch(error => {
            showError('Error analyzing image: ' + error.message);
        })
        .finally(() => {
            analyzeBtn.disabled = false;
        });
    });
    
    // Display prediction results
    function displayResults(results) {
        resultCategories.innerHTML = '';
        
        if (!results || results.length === 0) {
            showError('No results returned from the model');
            return;
        }
        
        // Find the predicted result (marked by is_predicted flag)
        const predictedResult = results.find(result => result.is_predicted) || results[0];
        
        // Update status
        resultStatus.textContent = `Analysis complete: ${predictedResult.class} detected`;
        
        // Create result category element
        const categoryEl = document.createElement('div');
        categoryEl.className = 'result-category highest-probability';
        
        categoryEl.innerHTML = `
            <span class="category-name">${predictedResult.class}</span>
        `;
        
        resultCategories.appendChild(categoryEl);
    }
    
    // Show error message
    function showError(message) {
        resultStatus.textContent = message;
        resultStatus.style.color = 'var(--accent-color)';
        
        // Reset after 3 seconds
        setTimeout(() => {
            resultStatus.style.color = 'var(--light-text)';
        }, 3000);
    }
    
    // Handle scroll events to update active navigation link
    window.addEventListener('scroll', function() {
        const scrollPosition = window.scrollY;
        
        document.querySelectorAll('section').forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionBottom = sectionTop + section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                const currentId = section.getAttribute('id');
                document.querySelectorAll('nav a').forEach(navLink => {
                    navLink.classList.remove('active');
                    if (navLink.getAttribute('href') === `#${currentId}`) {
                        navLink.classList.add('active');
                    }
                });
            }
        });
    });
});