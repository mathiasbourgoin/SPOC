document.addEventListener('DOMContentLoaded', () => {
  const themeToggle = document.getElementById('theme-toggle');
  const body = document.body;
  const icon = themeToggle.querySelector('span');

  // Check for saved theme preference or use system default
  const savedTheme = localStorage.getItem('theme');
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedTheme === 'dark' || (!savedTheme && systemDark)) {
    body.classList.add('dark-theme');
    icon.innerText = '☀️';
  } else {
    body.classList.remove('dark-theme');
    icon.innerText = '🌙';
  }

  themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-theme');
    const isDark = body.classList.contains('dark-theme');
    
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    icon.innerText = isDark ? '☀️' : '🌙';
  });
});
