// Initialize processor
const processor = new MaskProcessor(400, 400, 1.0, 20, 0.01);

// Control handlers
function updateControl(controlId, value, processorProperty, formatValue = v => v) {
    document.getElementById(`${controlId}Value`).textContent = formatValue(value);
    processor[processorProperty] = value;
    processor.processMask();
}

function updateLineLength(value) {
    updateControl('lineLength', value, 'lineLengthRatio', v => (v/1.0).toFixed(2));
}

function updateNumLines(value) {
    updateControl('numLines', value, 'numLines', v => parseInt(v));
}

function updateLineWidth(value) {
    updateControl('lineWidth', (value/1000).toFixed(3), 'lineWidthRatio', v => (v*1000));
}

function regenerateMask() {
    console.log('regenerateMask called');
    try {
        processor.generateRandomMask();
        processor.processMask();
    } catch (error) {
        console.error('Error in regenerateMask:', error);
    }
}

// Initial mask generation
console.log('Running initial mask generation');
regenerateMask();

// Drag and drop handlers
const dropZone = document.getElementById('dropZone');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
});

dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    
    // Remove any existing animation classes
    dropZone.classList.remove('error-flash');
    dropZone.classList.remove('success-flash');
    
    // Force a reflow to reset the animation
    void dropZone.offsetWidth;
    
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith('image/')) {
        dropZone.classList.add('error-flash');
        return;
    }

    try {
        const img = new Image();
        img.src = URL.createObjectURL(file);
        await img.decode();

        // Use the new loadMask method
        processor.loadMask(img);
        
        dropZone.classList.add('success-flash');
    } catch (error) {
        console.error('Error loading image:', error);
        dropZone.classList.add('error-flash');
    }
});

// Show drop zone on page load
dropZone.style.display = 'flex';

// Add download functionality
function downloadCanvas(type) {
    const button = event.currentTarget;
    const canvas = type === 'input' ? processor.domInputCanvas : processor.domOutputCanvas;
    
    // Create and trigger download
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `mask-${type}-${timestamp}.png`;
    link.href = canvas.toDataURL('image/png');
    
    // Add explosion animation
    button.classList.add('exploding');
    
    // Remove animation class after it completes
    button.addEventListener('animationend', () => {
        button.classList.remove('exploding');
    }, { once: true });
    
    link.click();
} 