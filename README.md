# ThoughtNebula

An in-browser, serverless demo showing how large language models map text into semantic space. Sentences are embedded with Transformers.js and projected with UMAP into 3D; each point lights up inside a stylized brain volume to illustrate how meaning clusters in vector space.

## Why it exists
- Give an intuitive feel for how LLM embeddings capture semantics without sharing data to a server.
- Show that “similar meaning = nearby vectors” by letting you search and see neighborhoods light up.
- Make the embedding process tangible with a brain-shaped spatial metaphor (neurons as points).

## How it works
- **Embedding**: Uses `EmbeddingGemma` via Transformers.js to turn each sentence into a high‑dimensional vector in the browser.
- **Projection**: Runs UMAP (3D) to compress embeddings into coordinates, then warps them into a brain-like ellipsoid so clusters look like neuron constellations.
- **Rendering**: React + `@react-three/fiber` + `@react-three/drei` + postprocessing for bloom; points glow based on similarity and search focus.
- **Search**: New queries are embedded client-side; cosine similarity reorders points and drives a focus animation.

## Running locally
```bash
npm install
npm run dev   # or npm run build && npm run preview
```
Open the dev server URL (typically http://localhost:5173). Everything runs locally—no backend required.

## Assets (brain model and music)
- Place your own brain GLB at `public/brain_hologram.glb`.
- Place your own background music at `public/music.mp3`.
These files are `.gitignore`d so you can use licensed assets without committing them. Ensure you have rights to the files you supply and credit their authors per their licenses.

### Deploying without committing assets
Host your assets somewhere accessible (e.g., object storage) and set env vars so the site can load them:
- `VITE_BRAIN_URL` → full URL to `brain_hologram.glb`
- `VITE_MUSIC_URL` → full URL to `music.mp3`

For GitHub Pages, add repo secrets `VITE_BRAIN_URL` and `VITE_MUSIC_URL`; the included workflow passes them into the build so the static site points to your hosted files. Locally, you can set these in a `.env.local` (ignored).

## Using the app
- Load the model (cached after first download).
- Enter full sentences, one per line. Short, natural sentences embed best.
- Generate to see points arranged in the brain volume.
- Search to highlight semantic neighbors; click points to refocus.
- Adjust quality presets if performance is low (battery saver → max fidelity).

## Design notes
- The brain mesh provides a familiar “neuron field” to house the projected vectors.
- Placeholder stars animate while the model loads, then swap to your embeddings.
- Camera framing is tuned differently for desktop vs mobile to keep labels readable.

## Credits
- Inspired by the Hugging Face Space: https://huggingface.co/spaces/webml-community/semantic-galaxy
- Brain model: “Arknights Sanity Brain” from Sketchfab https://sketchfab.com/3d-models/arknights-sanity-brain-a279a847a2d74cabbec1c75ea0ebc28e
- Music: "Mindwave" by Avanti — https://freetouse.com/music (Free No Copyright Music)
