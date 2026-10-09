// HTTP helpers for the backend (proxied by vite, see vite.config.js)

export async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function putJson(url, body) {
  const res = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`PUT ${url} failed: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function getBinary(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
  return res.arrayBuffer();
}
