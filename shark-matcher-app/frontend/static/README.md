# Frontend Static Assets

The single-page browser interface: HTML shell, stylesheets, images, and ES-module JavaScript, served by the app's own web server.

## Contents

| Path | Responsibility |
| --- | --- |
| `index.html` | Page shell; loads the stylesheets and `js/main.js` as a module and applies the saved light or dark theme. |
| `css/style.css` | Application styles; imports web fonts from Google Fonts. |
| `css/vendor/normalize.css` | Third-party CSS reset. |
| `js/main.js` | Entry point: loads `/api/runtime/config`, optionally loads the demo module, revalidates the saved session, then starts the UI. |
| `js/modules/shark_matcher/` | Browser API client: one module per endpoint family (`action`, `individual`, `match`, `media`, `proposal`, `review`, `user`), with `core.js` as the shared transport and local state and `base.js` as the public facade. |
| `js/modules/webui/` | User interface: `routes.js` maps location hashes to pages, and each page (`home`, `search`, `match`, `merge`, `upload`, `history`, `proposal`, `review`, `settings`, `login`, `individual`) has its own module, with `util.js`, `core.js`, and `log.js` (leveled console logging) for shared helpers. |
| `images/shark-loading.png` | Loading indicator. |
| `favicon.ico` | Tab icon. |
| `__init__.py` | Makes the folder a package so the server can read these files with `importlib.resources`. |

## Serving

`frontend/handlers.py` serves everything here under the `/static/` URL prefix, and the server redirects `/` to `/static/index.html`. Shark photos are not in this folder; they are served from the images directory in `core/app_constants.py` under `/static/frontend-images/`, after authentication.

Start the app as described in `scripts/README.md`, then open the printed address in a browser.

## Checks

Lint the JavaScript from the repository root:

```bash
python3 -m tests.tools.run_tests --js-style
```

The rules are in `frontend/eslint.config.js`, and `demo/*.js` is linted with the same rules. Browser workflows are exercised by `tests/browser/`.

## Boundaries

- No build step: files are served as written.
- The browser calls only the `/api/` endpoints routed in `frontend/server.py`; it never reads the database.
- Every page is a hash route inside the single page `index.html`.
- Python files here are limited to `__init__.py`.
