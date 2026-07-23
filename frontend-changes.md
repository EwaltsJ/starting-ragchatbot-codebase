# Frontend Changes: Theme Toggle Button

## Summary
Added a light/dark theme toggle button to the Course Materials Assistant UI, fitting the existing dark-first design aesthetic.

## Files Changed

### `frontend/index.html`
- Added a `#themeToggle` `<button>` inside `.container`, containing two inline SVG icons (sun and moon).
- Bumped cache-busting query params (`style.css?v=10`, `script.js?v=10`).

### `frontend/style.css`
- Added a `:root[data-theme="light"]` block that overrides all existing CSS custom properties (`--background`, `--surface`, `--text-primary`, `--border-color`, etc.) with a light palette, so every existing component restyles automatically without needing new selectors.
- Added `transition` rules (`background-color`, `border-color`, `color`, `box-shadow`) on `body` and the main themed containers/controls for a smooth crossfade when switching themes.
- Added `.theme-toggle` styles: a fixed-position circular button pinned to the top-right corner (`top: 1.25rem; right: 1.25rem`), using the same surface/border/shadow tokens as other controls (e.g. `#sendButton`), with hover lift and `:focus-visible` ring consistent with other interactive elements (`--focus-ring`).
- Added `.theme-icon` rules that crossfade/rotate between the sun and moon icons based on the current theme (opacity + rotate/scale transition), and a smaller mobile variant under the existing `@media (max-width: 768px)` breakpoint.

### `frontend/script.js`
- Added `themeToggle` to the cached DOM elements and wired a `click` listener in `setupEventListeners()`.
- Added `initTheme()`, called on load: reads a saved theme from `localStorage`, falling back to the OS `prefers-color-scheme` setting.
- Added `toggleTheme()`: flips between `light`/`dark` and persists the choice to `localStorage`.
- Added `applyTheme(theme)`: sets/removes `data-theme="light"` on `<html>` and updates `aria-pressed`/`aria-label` on the button so assistive tech announces the current state and the action the button performs.

## Behavior
- Default theme is dark (matches the original design); toggling switches to a light palette using the same component structure.
- Theme preference persists across reloads via `localStorage`, and respects the user's OS preference on first visit.
- The button is a native `<button>` element, so it's reachable via Tab and triggers via Enter/Space out of the box — no custom key handling needed.
- Icon swap and background/color changes animate smoothly (0.3–0.4s transitions) rather than snapping.

## Light Theme Palette Refinement

Revisited the `:root[data-theme="light"]` color values in `frontend/style.css` to make sure the light variant reads as an intentional, accessible palette rather than a straight inversion:

| Token | Value | Purpose |
|---|---|---|
| `--background` | `#f8fafc` | Page background (soft off-white, not stark white, to keep surface cards visually distinct) |
| `--surface` | `#ffffff` | Cards, message bubbles, inputs |
| `--surface-hover` | `#e2e8f0` | Hover state for pills/suggested items |
| `--text-primary` | `#0f172a` | Primary text — contrast ratio ~19:1 on `--background` |
| `--text-secondary` | `#52606d` | Secondary/meta text — contrast ratio ~6.2:1 on `--background` (passes WCAG AA) |
| `--border-color` | `#cbd5e1` | Card/input borders, visible but not heavy on light surfaces |
| `--primary-color` | `#1d4ed8` | Darkened from the dark-theme value (`#2563eb`) specifically for light backgrounds — contrast ratio ~6.7:1 on white (dark theme's lighter blue would only hit ~4.9:1) |
| `--primary-hover` | `#1e40af` | Hover/active state for the primary color |
| `--user-message` | `#1d4ed8` | Matches the adjusted primary so user chat bubbles keep sufficient contrast with white text |
| `--focus-ring` | `rgba(29, 78, 216, 0.3)` | Focus outline, tuned to the adjusted primary |

All existing components (sidebar, suggested-question buttons, source pills, course stats, chat bubbles, inputs) restyle automatically since they reference these shared custom properties — no component-specific CSS was needed. Verified visually in a browser (light-mode sidebar, expanded course list, and suggested questions) that text stays legible and borders remain visible against the light backgrounds.
