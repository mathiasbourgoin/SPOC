document.addEventListener('DOMContentLoaded', () => {
  const highlightSarek = () => {
    const codeBlocks = document.querySelectorAll('pre code, .highlight pre');
    
    // Sarek specific keywords and extensions
    const patterns = [
      { regex: /%(kernel|shared|superstep)/g, class: 'k' },
      { regex: /@sarek\.(module|type)/g, class: 'k' },
      { regex: /@@sarek\.type/g, class: 'k' },
      { regex: /\b(mut)\b/g, class: 'k' } // Sarek mutable keyword
    ];

    codeBlocks.forEach(block => {
      let html = block.innerHTML;
      let modified = false;

      patterns.forEach(item => {
        // We use a simple replacement. To avoid matching inside existing HTML tags, 
        // we'd need a more complex parser, but for these specific tokens in OCaml blocks,
        // it's generally safe.
        if (item.regex.test(html)) {
          html = html.replace(item.regex, `<span class="${item.class}">$1</span>`);
          modified = true;
        }
      });

      if (modified) {
        block.innerHTML = html;
      }
    });
  };

  highlightSarek();
  
  // Also run after Thebe activation if possible
  // Since Thebe uses CodeMirror, it has its own highlighting, 
  // but for the static parts, this works well.
});
