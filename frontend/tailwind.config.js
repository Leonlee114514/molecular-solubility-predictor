/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        /* ============================================================
         * Token layer. Values live in src/index.css (:root); this file only
         * maps names. Never hardcode a hex here — if a colour is missing,
         * add it to :root first.
         * ============================================================ */

        /* surface ladder + ink ladder + hairline */
        ob: {
          bg: 'hsl(var(--ob-bg) / <alpha-value>)',
          surface: 'hsl(var(--ob-surface) / <alpha-value>)',
          sunken: 'hsl(var(--ob-sunken) / <alpha-value>)',
          well: 'hsl(var(--ob-well) / <alpha-value>)',
          rule: 'hsl(var(--ob-rule) / <alpha-value>)',
          text: 'hsl(var(--ob-text) / <alpha-value>)',
          muted: 'hsl(var(--ob-muted) / <alpha-value>)',
          faint: 'hsl(var(--ob-faint) / <alpha-value>)',
          // kept so the 33 existing `border-ob-border` call sites resolve
          border: 'hsl(var(--ob-rule) / <alpha-value>)',
        },

        /* the only chrome hue */
        brand: {
          DEFAULT: 'hsl(var(--brand) / <alpha-value>)',
          strong: 'hsl(var(--brand-strong) / <alpha-value>)',
          soft: 'hsl(var(--brand-soft) / <alpha-value>)',
        },

        /* verdict colours — meaning only, never decoration */
        ok: 'hsl(var(--ok) / <alpha-value>)',
        warn: 'hsl(var(--warn) / <alpha-value>)',
        danger: 'hsl(var(--danger) / <alpha-value>)',

        /* ---- compatibility aliases (do not use in new code) ---- */
        // The previous theme's names, now resolving to brand/warn. Kept so the
        // ~40 existing call sites keep working until batch 2 rewrites them.
        nebula: {
          DEFAULT: 'hsl(var(--brand) / <alpha-value>)',
          light: 'hsl(var(--brand-soft) / <alpha-value>)',
        },
        stargold: 'hsl(var(--warn) / <alpha-value>)',
        orbit: 'hsl(var(--ob-muted) / <alpha-value>)',

        /* ============================================================
         * Transitional bridge — TEMPORARY, delete in batch 2.
         *
         * Components still colour callouts with Tailwind's stock palette
         * (emerald/amber/cyan/red/blue). Mapping the few shades they use onto
         * the verdict palette collapses the result page from four hues onto
         * ink-blue + ok/warn/danger without touching component code.
         *
         * Only the shades that are actually referenced are redefined. Any
         * other shade would silently keep Tailwind's stock green/amber/cyan,
         * so do not add new usages — replace them with ok/warn/danger/brand.
         * ============================================================ */
        emerald: {
          500: '#2f7d4f', // tint / border only, never text
          700: '#14743a', // = ok
        },
        amber: {
          500: '#b07a1e', // tint / border only, never text
          700: '#8f5d0a', // = warn
          800: '#7a4f08', // deeper warn text
        },
        cyan: {
          500: '#2c5282', // folded into brand-soft: kills the fifth hue
        },
        red: {
          500: '#c0392b', // tint / border only
          700: '#b91c1c', // = danger
        },
        blue: {
          400: '#2c5282', // folded into brand-soft
        },

        /* shadcn bridge (ui/ components) */
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive) / <alpha-value>)',
          foreground: 'hsl(var(--destructive-foreground) / <alpha-value>)',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        sidebar: {
          DEFAULT: 'hsl(var(--sidebar-background))',
          foreground: 'hsl(var(--sidebar-foreground))',
          primary: 'hsl(var(--sidebar-primary))',
          'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
          accent: 'hsl(var(--sidebar-accent))',
          'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
          border: 'hsl(var(--sidebar-border))',
          ring: 'hsl(var(--sidebar-ring))',
        },
      },

      /* 4px base; card 3px, control 4px, panel 6px. The old xl/lg/md/sm/xs
         names stay because components use them, but they now resolve to three
         visually distinct steps instead of five near-identical ones. */
      borderRadius: {
        xl: 'calc(var(--radius) + 2px)',
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 1px)',
        sm: 'calc(var(--radius) - 2px)',
        xs: 'calc(var(--radius) - 3px)',
        card: 'calc(var(--radius) - 1px)',
        control: 'var(--radius)',
        panel: 'calc(var(--radius) + 2px)',
      },

      boxShadow: {
        hairline: 'var(--shadow-hairline)',
        pop: 'var(--shadow-pop)',
        // legacy names -> the report elevation scale
        xs: '0 1px 0 hsl(36 10% 10% / 0.04)',
        sm: '0 1px 2px hsl(36 10% 10% / 0.06)',
        md: 'var(--shadow-pop)',
        lg: 'var(--shadow-pop)',
        glow: 'var(--shadow-hairline)',
      },

      /* One scale, explicitly declared. Values match Tailwind's defaults for
         xs..3xl so nothing reflows; only the display sizes get tighter
         tracking, which is what makes the head read as a masthead. */
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.01em' }],
        xs: ['0.75rem', { lineHeight: '1rem' }],
        sm: ['0.875rem', { lineHeight: '1.25rem' }],
        base: ['1rem', { lineHeight: '1.5rem' }],
        lg: ['1.125rem', { lineHeight: '1.75rem' }],
        xl: ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem', letterSpacing: '-0.02em' }],
        '5xl': ['3rem', { lineHeight: '3.25rem', letterSpacing: '-0.03em' }],
        '6xl': ['3.75rem', { lineHeight: '3.75rem', letterSpacing: '-0.035em' }],
      },

      fontFamily: {
        sans: [
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'PingFang SC',
          'Hiragino Sans GB',
          'Microsoft YaHei',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'SF Mono',
          'Menlo',
          'Consolas',
          'Liberation Mono',
          'monospace',
        ],
      },

      /* semantic spacing on the same 4px grid (batch 3 uses these) */
      spacing: {
        gutter: '1rem',
        section: '2rem',
        page: '2.5rem',
      },

      transitionDuration: {
        fast: '120ms',
        base: '180ms',
        slow: '240ms',
      },

      transitionTimingFunction: {
        out: 'cubic-bezier(0.2, 0.8, 0.2, 1)',
        inout: 'cubic-bezier(0.4, 0, 0.2, 1)',
      },

      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'caret-blink': {
          '0%,70%,100%': { opacity: '1' },
          '20%,50%': { opacity: '0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s var(--ease-out)',
        'accordion-up': 'accordion-up 0.2s var(--ease-out)',
        'caret-blink': 'caret-blink 1.25s ease-out infinite',
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
