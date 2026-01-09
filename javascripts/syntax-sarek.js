document.addEventListener('DOMContentLoaded', () => {
  const highlightSarek = () => {
    // Target both standard code blocks and Jekyll's highlighted blocks
    const codeBlocks = document.querySelectorAll('pre code, .highlight pre, .tab-content pre');
    
    codeBlocks.forEach(block => {
      let html = block.innerHTML;
      
      // 1. Sarek OCaml Extensions
      // Matches %kernel, %shared, %superstep even if separated by tags
      html = html.replace(/%(kernel|shared|superstep)/g, '<span class="k">%$1</span>');
      
      // Matches attributes like [@sarek.module] or [@@sarek.type]
      html = html.replace(/@sarek\.(module|type)/g, '<span class="k">@sarek.$1</span>');
      html = html.replace(/@@sarek\.type/g, '<span class="k">@@sarek.type</span>');
      
      // 2. Sarek OCaml Keywords
      // Highlighting 'mut' as a keyword
      html = html.replace(/\b(mut)\b/g, '<span class="k">$1</span>');

      // 3. OpenCL Keywords (since we use 'c' highlighting as a base)
      const openClKeywords = ['__kernel', '__global', '__local', 'get_global_id', 'get_local_id', 'get_group_id', 'get_local_size', 'barrier', 'CLK_LOCAL_MEM_FENCE', 'CLK_GLOBAL_MEM_FENCE'];
      openClKeywords.forEach(kw => {
        const reg = new RegExp(`\b(${kw})\b`, 'g');
        html = html.replace(reg, '<span class="k">$1</span>');
      });

      // 4. Metal Keywords
      const metalKeywords = ['kernel', 'device', 'threadgroup', 'thread_position_in_grid', 'thread_index_in_threadgroup'];
      metalKeywords.forEach(kw => {
        const reg = new RegExp(`\b(${kw})\b`, 'g');
        html = html.replace(reg, '<span class="k">$1</span>');
      });

      block.innerHTML = html;
    });
  };

  // Run immediately
  highlightSarek();
  
  // Re-run when tabs are switched to ensure visibility
  document.querySelectorAll('.tab-header').forEach(tab => {
    tab.addEventListener('click', () => {
      setTimeout(highlightSarek, 10);
    });
  });
});