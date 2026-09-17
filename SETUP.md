# FloodReroute frontend — setup

## 1. Scaffold

```bash
npm create vite@latest floodreroute-web -- --template react-ts
cd floodreroute-web
npm install
npm install react-map-gl maplibre-gl
npm install -D tailwindcss @tailwindcss/vite @types/geojson
```

## 2. Enable Tailwind v4

`vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({ plugins: [react(), tailwindcss()] });
```

## 3. Drop in the files

Replace `src/index.css`, `src/App.tsx`, and add `src/api.ts`,
`src/FloodMap.tsx`, `src/RoutePanel.tsx`.

Delete `src/App.css`. In `src/main.tsx` make sure the import is
`import "./index.css"`.

## 4. Fonts

Add to `index.html` inside `<head>`:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
```

## 5. Point at the backend

`.env.local`:

```
VITE_API_URL=http://127.0.0.1:8000
```

## 6. Backend CORS

`main.py` reads ALLOWED_ORIGINS from the environment and defaults to
localhost:5173 and :3000, so Vite's default port already works. For
Vercel later:

```bash
ALLOWED_ORIGINS=https://your-app.vercel.app
```

## 7. Run

```bash
npm run dev
```

Backend on :8000, frontend on :5173.

## Switching to Mapbox

Currently MapLibre with free OSM raster tiles — no token needed.
For Mapbox:

```bash
npm install mapbox-gl
```

In `FloodMap.tsx`: import from `react-map-gl/mapbox`, swap the CSS
import to `mapbox-gl/dist/mapbox-gl.css`, drop the inline `STYLE`
object, and pass `mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}`
plus `mapStyle="mapbox://styles/mapbox/dark-v11"`.
