document.addEventListener('DOMContentLoaded', () => {
  // Find all code blocks with class "language-mermaid"
  const mermaidBlocks = document.querySelectorAll('pre code.language-mermaid');
  
  mermaidBlocks.forEach(block => {
    const pre = block.parentElement;
    const code = block.innerText;
    
    // Create a new div with class "mermaid"
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.innerText = code;
    
    // Replace the pre element with the new div
    pre.replaceWith(div);
  });

  // Now that the blocks are transformed, we can initialize Mermaid if not already done
  // or trigger a re-render. Since we are using ES modules, we should handle this 
  // in the main layout or here if we import it.
});
