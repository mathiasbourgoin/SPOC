document.addEventListener('DOMContentLoaded', () => {
  const highlightSarek = () => {
    // Target code blocks
    const codeBlocks = document.querySelectorAll('pre code, .highlight pre, .tab-content pre');
    
    codeBlocks.forEach(block => {
      let html = block.innerHTML;
      
      // 1. Handle Sarek OCaml Extensions that might be split by Rouge
      // Example: <span class="o">%</span><span class="n">kernel</span>
      const extensions = ['kernel', 'shared', 'superstep'];
      extensions.forEach(ext => {
        // Match % optionally followed by HTML tags, then the extension name
        const reg = new RegExp('%(?:<[^>]+>)*' + ext, 'g');
        html = html.replace(reg, (match) => `<span class="k">${match}</span>`);
      });

      // 2. Handle Sarek attributes
      const attributes = [/@sarek\.module/g, /@sarek\.type/g, /@@sarek\.type/g];
      attributes.forEach(attr => {
        html = html.replace(attr, (match) => `<span class="k">${match}</span>`);
      });

      // 3. Keywords
      const keywords = ['mut', '__kernel', '__global', '__local', 'get_global_id', 'get_local_id', 'barrier'];
      keywords.forEach(kw => {
        // Use a negative lookbehind/lookahead to avoid matching inside existing tags or partial words
        const reg = new RegExp('\b' + kw + '\b', 'g');
        // Simple check to avoid double-wrapping
        if (!html.includes(`<span class="k">${kw}</span>`)) {
           html = html.replace(reg, `<span class="k">${kw}</span>`);
        }
      });

      block.innerHTML = html;
    });
  };

  highlightSarek();
  
  // Re-run on tab switch
  document.querySelectorAll('.tab-header').forEach(tab => {
    tab.addEventListener('click', () => {
      setTimeout(highlightSarek, 50);
    });
  });

  // Re-run when Thebe is activated
  const thebeBtn = document.getElementById('thebe-activate');
  if (thebeBtn) {
    thebeBtn.addEventListener('click', () => {
      // Periodic check for 10 seconds as Thebe loads
      let checks = 0;
      const interval = setInterval(() => {
        highlightSarek();
        if (checks++ > 20) clearInterval(interval);
      }, 500);
    });
  }
});
