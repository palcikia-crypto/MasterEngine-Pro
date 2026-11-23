/* MasterEngine service worker — offline + instant loads + auto-updates. */
const VERSION = "me-v4";
const BASE = new URL("./", self.location.href);
const ASSETS = [
  "./",
  "index.html",
  "app.js",
  "engine-worker.js",
  "js-engine.js",
  "manifest.webmanifest",
  "icon-192.png",
  "icon-512.png",
  "apple-touch-icon.png",
].map((a) => new URL(a, BASE).href);

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    await Promise.allSettled(ASSETS.map((a) => cache.add(new Request(a, { cache: "reload" }))));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== BASE.origin) return;

  e.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const cached = await cache.match(req, { ignoreSearch: url.pathname === "./" || url.pathname.endsWith("index.html") });
    const net = fetch(req).then((resp) => {
      if (resp && resp.ok) cache.put(req, resp.clone()).catch(() => {});
      return resp;
    }).catch(() => null);
    if (cached) {
      net.catch(() => {});
      return cached;
    }
    const resp = await net;
    if (resp) return resp;
    return new Response("MasterEngine: offline and not cached", { status: 503, statusText: "Offline" });
  })());
});
