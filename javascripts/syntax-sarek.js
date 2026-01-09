document.addEventListener('DOMContentLoaded', () => {
  const highlightSarek = () => {
    // Target code blocks
    const codeBlocks = document.querySelectorAll('pre code, .highlight pre, .tab-content pre');
    
    codeBlocks.forEach(block => {
      let html = block.innerHTML;
      
      // Fix for word boundary regex: need double backslash in constructor
      const wrapKeyword = (regex, text) => {
        // Simple replacement but careful with already highlighted spans
        html = html.replace(regex, `<span class="k">${text}</span>`);
      };

      // 1. Sarek OCaml Extensions
      html = html.replace(/%(kernel|shared|superstep)/g, '<span class="k">%$1</span>');
      html = html.replace(/@sarek\.(module|type)/g, '<span class="k">@sarek.$1</span>');
      html = html.replace(/@@sarek\.type/g, '<span class="k">@@sarek.type</span>');
      
      // 2. Sarek OCaml Keywords
      html = html.replace(/\bmut\b/g, '<span class="k">mut</span>');

      // 3. OpenCL / CUDA / Metal Keywords
      const gpuKeywords = [
        '__kernel', '__global', '__local', 'get_global_id', 'get_local_id', 
        'get_group_id', 'get_local_size', 'barrier', 'CLK_LOCAL_MEM_FENCE', 
        'CLK_GLOBAL_MEM_FENCE', 'thread_idx_x', 'block_idx_x', 'block_dim_x',
        'device', 'threadgroup', 'thread_position_in_grid'
      ];
      
      gpuKeywords.forEach(kw => {
        const reg = new RegExp(`\\b${kw}\\b`, 'g');
        html = html.replace(reg, `<span class="k">${kw}</span>`);
      });

      block.innerHTML = html;
    });
  };

  // Initial Run
  highlightSarek();
  
  // Re-run on tab switch
  document.querySelectorAll('.tab-header').forEach(tab => {
    tab.addEventListener('click', () => {
      setTimeout(highlightSarek, 50);
    });
  });
});
