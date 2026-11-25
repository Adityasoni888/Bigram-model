# Token Observatory

A standalone language-model learning studio built with plain HTML, CSS, and JavaScript. It preserves the core learning activities of the supplied reference—sequence perplexity, next-token predictions, text generation, sentence comparison, a hand-set probability sandbox, and an explanatory guide—inside a new interface and product identity.

## Run locally

From this directory, start any static file server. For example:

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a modern browser. The application has no build step or application server. On first use, it loads Transformers.js and the quantized DistilGPT-2 model from public CDNs/model hosting; that requires internet access and the model is cached by the browser. The manual probability sandbox, guide, appearance toggle, session log, search, and CSV export work without loading the model.

## Project files

- `index.html` — page structure and content
- `styles.css` — responsive visual system
- `app.js` — model inference, controls, probability math, and browser-local session history

Text analyzed in the app is passed to the model running in the browser. Session history is stored in local storage on this device.
