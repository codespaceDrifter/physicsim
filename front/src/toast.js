// Floating notification at the bottom of the window. kind = 'error' | 'info'
const box = document.getElementById('toasts');

export function toast(message, kind = 'error') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  box.append(el);
  setTimeout(() => el.remove(), 5000);
}
