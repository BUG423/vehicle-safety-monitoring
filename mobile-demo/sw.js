/* 离线缓存 —— 让演示页在无网络时也能打开（车载场景本身就常断网，演示也该如此）。 */
const CACHE = 'vsm-demo-v1';
const ASSETS = ['./', './index.html', './mode-a.html', './mode-b.html', './mode-c.html',
                './manifest.json', './assets/icon-192.png', './assets/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE)
    .then((c) => Promise.allSettled(ASSETS.map((u) => c.add(u))))   // 个别页缺失不应导致整体失败
    .then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  // 网络优先、离线回落缓存：保证演示页更新后能拿到新版本
  e.respondWith(
    fetch(e.request)
      .then((r) => {
        const copy = r.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return r;
      })
      .catch(() => caches.match(e.request).then((r) => r || caches.match('./index.html')))
  );
});
