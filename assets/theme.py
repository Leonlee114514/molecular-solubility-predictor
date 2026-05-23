"""
DisSolve - Deep Space Molecular Universe Theme CSS.
"""

import streamlit as st

THEME_CSS = """<style>

/* ═══════════════════════════════════════════════
   DisSolve — Deep Space Molecular Universe Theme
   ═══════════════════════════════════════════════ */

/* Google Fonts removed — system font stack for China accessibility */
/* ─── 0. 粒子背景动画 ─── */
@keyframes nebulaPulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.1); }
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(124, 58, 237, 0.1); }
    50% { box-shadow: 0 0 40px rgba(124, 58, 237, 0.2); }
}

@keyframes starTwinkle {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

@keyframes molecularDiffuse {
    from { opacity: 0; transform: translateY(24px) scale(0.96); filter: blur(2px); }
    to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}

@keyframes electronJump {
    0% { transform: translateX(-100%); opacity: 0; }
    50% { opacity: 1; }
    100% { transform: translateX(400%); opacity: 0; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ─── 1. 核心变量 ─── */
:root {
    --ob-bg-primary: #131328;
    --ob-bg-surface: #1a1a2e;
    --ob-bg-elevated: rgba(30, 30, 46, 0.6);
    --ob-nebula: #7c3aed;
    --ob-nebula-light: #a78bfa;
    --ob-nebula-glow: rgba(124, 58, 237, 0.15);
    --ob-star-gold: #fbbf24;
    --ob-orbit-cyan: #06b6d4;
    --ob-text-primary: #f0f0f5;
    --ob-text-secondary: #a0a0b0;
    --ob-text-tertiary: #6b6b7b;
    --ob-border: rgba(255, 255, 255, 0.08);
    --ob-border-hover: rgba(124, 58, 237, 0.3);
    --ob-radius: 16px;
    --ob-radius-sm: 12px;

    /* Base Web 暗色主题变量覆盖（下拉菜单、弹出框等） */
    --colors-menu-background-color: #1a1a2e;
    --colors-menu-color: #e0e0e0;
    --colors-menu-item-hover-background-color: rgba(124, 58, 237, 0.25);
    --colors-menu-item-selected-background-color: rgba(124, 58, 237, 0.35);
    --colors-popover-background-color: #1a1a2e;
    --colors-dropdown-menu-background-color: #1a1a2e;
    --colors-dropdown-menu-hover-background-color: rgba(124, 58, 237, 0.25);
    --colors-dropdown-menu-selected-background-color: rgba(124, 58, 237, 0.35);
}

/* ─── 2. 全局基础 ─── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .stApp {
    background: var(--ob-bg-primary) !important;
    background-color: var(--ob-bg-primary) !important;
    font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    color: var(--ob-text-primary);
}

.stApp {
    background: transparent !important;
    background-color: transparent !important;
    position: relative;
    min-height: 100vh;
}

/* 粒子星空层 - 密集多层 */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        /* === 大亮点（紫/青/金，模拟亮星）=== */
        radial-gradient(3.5px 3.5px at 8% 12%, rgba(196, 181, 253, 1.0), transparent),
        radial-gradient(3px 3px at 22% 8%, rgba(103, 232, 249, 0.95), transparent),
        radial-gradient(3.5px 3.5px at 35% 25%, rgba(167, 139, 250, 1.0), transparent),
        radial-gradient(2.5px 2.5px at 48% 5%, rgba(253, 224, 71, 0.9), transparent),
        radial-gradient(3.5px 3.5px at 62% 18%, rgba(34, 211, 238, 1.0), transparent),
        radial-gradient(3px 3px at 78% 10%, rgba(196, 181, 253, 1.0), transparent),
        radial-gradient(3.5px 3.5px at 88% 28%, rgba(103, 232, 249, 0.95), transparent),
        radial-gradient(2.5px 2.5px at 95% 8%, rgba(233, 213, 255, 0.9), transparent),
        radial-gradient(3.5px 3.5px at 15% 40%, rgba(34, 211, 238, 1.0), transparent),
        radial-gradient(3px 3px at 55% 35%, rgba(167, 139, 250, 0.95), transparent),
        radial-gradient(3.5px 3.5px at 72% 42%, rgba(253, 224, 71, 0.85), transparent),
        radial-gradient(2.5px 2.5px at 42% 48%, rgba(233, 213, 255, 0.9), transparent),
        /* === 中等星点 === */
        radial-gradient(2.5px 2.5px at 5% 55%, rgba(139, 92, 246, 0.9), transparent),
        radial-gradient(2px 2px at 18% 62%, rgba(103, 232, 249, 0.85), transparent),
        radial-gradient(2.5px 2.5px at 28% 75%, rgba(196, 181, 253, 0.9), transparent),
        radial-gradient(2px 2px at 40% 68%, rgba(34, 211, 238, 0.8), transparent),
        radial-gradient(2.5px 2.5px at 52% 82%, rgba(167, 139, 250, 0.85), transparent),
        radial-gradient(2px 2px at 65% 58%, rgba(253, 224, 71, 0.75), transparent),
        radial-gradient(2.5px 2.5px at 75% 72%, rgba(139, 92, 246, 0.9), transparent),
        radial-gradient(2px 2px at 85% 55%, rgba(103, 232, 249, 0.85), transparent),
        radial-gradient(2.5px 2.5px at 92% 78%, rgba(233, 213, 255, 0.8), transparent),
        radial-gradient(2px 2px at 12% 88%, rgba(34, 211, 238, 0.75), transparent),
        radial-gradient(2.5px 2.5px at 32% 92%, rgba(167, 139, 250, 0.85), transparent),
        radial-gradient(2px 2px at 58% 95%, rgba(196, 181, 253, 0.8), transparent),
        radial-gradient(2.5px 2.5px at 82% 88%, rgba(103, 232, 249, 0.75), transparent),
        /* === 白色小星点 === */
        radial-gradient(2px 2px at 3% 30%, rgba(255, 255, 255, 0.8), transparent),
        radial-gradient(1.5px 1.5px at 12% 18%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(2px 2px at 25% 45%, rgba(255, 255, 255, 0.8), transparent),
        radial-gradient(1.5px 1.5px at 38% 15%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(2px 2px at 45% 58%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(1.5px 1.5px at 58% 12%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(2px 2px at 68% 50%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(1.5px 1.5px at 82% 35%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(2px 2px at 90% 60%, rgba(255, 255, 255, 0.75), transparent),
        radial-gradient(1.5px 1.5px at 8% 78%, rgba(255, 255, 255, 0.65), transparent),
        radial-gradient(2px 2px at 48% 88%, rgba(255, 255, 255, 0.7), transparent),
        radial-gradient(1.5px 1.5px at 70% 92%, rgba(255, 255, 255, 0.65), transparent),
        /* === 微星点（增加密度）=== */
        radial-gradient(1.5px 1.5px at 7% 5%, rgba(196, 181, 253, 0.65), transparent),
        radial-gradient(1.5px 1.5px at 30% 3%, rgba(103, 232, 249, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 50% 20%, rgba(233, 213, 255, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 60% 8%, rgba(167, 139, 250, 0.65), transparent),
        radial-gradient(1.5px 1.5px at 80% 48%, rgba(34, 211, 238, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 95% 42%, rgba(196, 181, 253, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 20% 52%, rgba(253, 224, 71, 0.55), transparent),
        radial-gradient(1.5px 1.5px at 45% 72%, rgba(196, 181, 253, 0.6), transparent),
        radial-gradient(1.5px 1.5px at 88% 68%, rgba(103, 232, 249, 0.55), transparent);
    background-size: 100% 100%;
    animation: starTwinkle 4s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 1.0;
}

/* 星云光晕叠加 - 多层 */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        /* 主星云：紫色弥散光（左上区域） */
        radial-gradient(ellipse 65% 50% at 25% 35%, rgba(124, 58, 237, 0.35) 0%, rgba(76, 29, 149, 0.15) 35%, transparent 65%),
        /* 副星云：青色弥散光（右上区域） */
        radial-gradient(ellipse 55% 45% at 75% 25%, rgba(6, 182, 212, 0.25) 0%, rgba(8, 145, 178, 0.1) 40%, transparent 70%),
        /* 底部暖光晕 */
        radial-gradient(ellipse 55% 40% at 50% 90%, rgba(251, 191, 36, 0.15) 0%, transparent 55%),
        /* 中央微弱紫光 */
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(109, 40, 217, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: nebulaPulse 10s ease-in-out infinite;
}

.main .block-container {
    background: transparent !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
    position: relative;
    z-index: 1;
}

code, pre, .mono {
    font-family: 'Cascadia Code', 'Consolas', 'Menlo', monospace !important;
}

/* ─── 3. 排版 ─── */
.gradient-title {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-weight: 700;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    text-align: center;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 30%, #06b6d4 70%, #22d3ee 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 8s ease infinite;
    filter: drop-shadow(0 0 20px rgba(124, 58, 237, 0.3));
}

.subtitle {
    text-align: center;
    color: var(--ob-text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
    margin-bottom: 2rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
    letter-spacing: 0.02em;
}

.tagline {
    text-align: center;
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 0.75rem;
    color: var(--ob-nebula-light);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.card-title {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    color: var(--ob-text-primary);
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.01em;
}

.card-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.2em;
    background: linear-gradient(180deg, var(--ob-nebula), var(--ob-orbit-cyan));
    border-radius: 4px;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
}

/* ─── 4. 卡片容器 ─── */
.card-container {
    background: linear-gradient(155deg, rgba(35, 35, 55, 0.55) 0%, rgba(20, 20, 35, 0.45) 50%, rgba(30, 30, 50, 0.40) 100%);
    backdrop-filter: blur(12px) saturate(120%);
    -webkit-backdrop-filter: blur(12px) saturate(120%);
    border: 1px solid var(--ob-border);
    border-radius: var(--ob-radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow:
        0 1px 2px rgba(0,0,0,0.2),
        0 4px 12px -2px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.15s ease;
    will-change: transform, box-shadow;
    position: relative;
    overflow: hidden;
    animation: molecularDiffuse 0.35s ease both;
}

.card-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(124, 58, 237, 0.35) 20%, rgba(6, 182, 212, 0.2) 50%, rgba(124, 58, 237, 0.35) 80%, transparent 100%);
    pointer-events: none;
}

.card-container::after {
    content: '';
    position: absolute;
    top: -40%; right: -20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.06) 0%, transparent 70%);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.card-container:hover {
    transform: translateY(-4px);
    box-shadow:
        0 20px 40px -8px rgba(0,0,0,0.4),
        0 0 0 1px rgba(124, 58, 237, 0.15),
        0 0 40px rgba(124, 58, 237, 0.08),
        inset 0 1px 0 rgba(255,255,255,0.06);
    border-color: rgba(124, 58, 237, 0.2);
}

.card-container:hover::after { opacity: 1; }

.card-container:nth-child(1) { animation-delay: 0.05s; }
.card-container:nth-child(2) { animation-delay: 0.10s; }
.card-container:nth-child(3) { animation-delay: 0.15s; }
.card-container:nth-child(4) { animation-delay: 0.20s; }

/* ─── 5. 按钮 ─── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #7c3aed, #8b5cf6) !important;
    background-size: 200% 200% !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--ob-radius-sm) !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 15px -3px rgba(124, 58, 237, 0.3), 0 0 0 0 rgba(124, 58, 237, 0) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-position 0.3s ease !important;
    will-change: transform, box-shadow;
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    transition: left 0.4s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 25px -3px rgba(0,0,0,0.3), 0 0 25px rgba(124, 58, 237, 0.2) !important;
    background-position: 100% 0% !important;
}

.stButton > button:hover::after { left: 100%; }

.stButton > button:active {
    transform: translateY(0) scale(0.97);
    box-shadow: 0 2px 8px -1px rgba(0,0,0,0.2), inset 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #0e7490, #06b6d4, #22d3ee) !important;
    box-shadow: 0 4px 15px -3px rgba(6, 182, 212, 0.3) !important;
}

.stButton > button[kind="secondary"]:hover {
    box-shadow: 0 10px 25px -3px rgba(0,0,0,0.3), 0 0 25px rgba(6, 182, 212, 0.2) !important;
}

/* ─── 6. 输入框 ─── */
/* 先彻底清除所有外层容器的默认白边/灰边 */
.stTextInput > div,
.stTextInput > div > div,
.stTextInput [data-baseweb="input"],
.stTextInput [data-baseweb="input"] > div,
.stTextInput [data-baseweb="input"] > div > div,
.stTextInput [data-testid="stTextInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 真正的 input 输入框样式（与 selectbox 统一） */
.stTextInput > div > div > input,
.stTextInput > div > div > textarea,
.stTextInput [data-baseweb="input"] > div,
.stTextInput [data-baseweb="input"] input {
    border-radius: var(--ob-radius-sm) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 0.625rem 1rem !important;
    font-size: 0.9375rem !important;
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.9) 0%, rgba(15, 15, 25, 0.7) 100%) !important;
    color: var(--ob-text-primary) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
}

.stTextInput > div > div > input::placeholder,
.stTextInput [data-baseweb="input"] input::placeholder {
    color: var(--ob-text-tertiary) !important;
}

.stTextInput > div > div > input:focus,
.stTextInput [data-baseweb="input"] > div:focus-within {
    border-color: var(--ob-nebula) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15), inset 0 1px 3px rgba(0,0,0,0.2) !important;
    outline: none !important;
}

/* ─── Selectbox ─── */
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div > div,
.stSelectbox [data-baseweb="select"] input {
    border-radius: var(--ob-radius-sm) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.9) 0%, rgba(15, 15, 25, 0.7) 100%) !important;
    color: var(--ob-text-primary) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(124, 58, 237, 0.35) !important;
}

.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--ob-nebula) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15), inset 0 1px 3px rgba(0,0,0,0.2) !important;
}

/* 清除 selectbox 原生白色背景 */
.stSelectbox svg,
.stSelectbox [data-baseweb="select"] svg {
    color: var(--ob-text-secondary) !important;
}

/* ─── 下拉菜单容器：全局强制覆盖（不局限在 .stSelectbox 内部） ─── */
/* Streamlit 的下拉菜单位于 body 层级的 portal/overlay 中 */
html body [data-baseweb="menu"],
html body [data-baseweb="popover"],
html body [data-baseweb="select"] [data-baseweb="menu"],
html body div[role="listbox"],
html body ul[role="listbox"] {
    background: #1a1a2e !important;
    background-color: #1a1a2e !important;
    background-image: none !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: var(--ob-radius-sm) !important;
    box-shadow: 0 10px 30px -5px rgba(0,0,0,0.4) !important;
    color: #e0e0e0 !important;
}

/* 菜单内直接子容器 —— 必须显式设回深色，覆盖浏览器/组件默认白底 */
html body [data-baseweb="menu"] > div,
html body [data-baseweb="menu"] > div > div:first-child,
html body [data-baseweb="menu"] [role="listbox"],
html body [data-baseweb="menu"] ul,
html body div[role="listbox"] > div,
html body ul[role="listbox"] > li {
    background: #1a1a2e !important;
    background-color: #1a1a2e !important;
    background-image: none !important;
    color: #e0e0e0 !important;
}

/* 下拉选项行 */
html body [role="option"],
html body [data-baseweb="menu"] [role="option"],
html body [data-baseweb="menu"] div[role="option"],
html body ul li[role="option"],
html body div[role="listbox"] div {
    color: #e0e0e0 !important;
    background: transparent !important;
    background-color: transparent !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

html body [role="option"]:hover,
html body [data-baseweb="menu"] [role="option"]:hover,
html body div[role="listbox"] div:hover {
    background: rgba(124, 58, 237, 0.25) !important;
    background-color: rgba(124, 58, 237, 0.25) !important;
    color: #ffffff !important;
}

html body [aria-selected="true"],
html body [data-baseweb="menu"] [aria-selected="true"],
html body div[role="listbox"] [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.35) !important;
    background-color: rgba(124, 58, 237, 0.35) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ─── 7. Metric ─── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 30, 46, 0.55) 0%, rgba(15, 15, 25, 0.4) 100%);
    border: 1px solid var(--ob-border);
    border-radius: var(--ob-radius-sm);
    padding: 1rem 1.25rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 2px 10px rgba(0,0,0,0.15);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(124, 58, 237, 0.2);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 4px 20px rgba(0,0,0,0.2), 0 0 20px rgba(124, 58, 237, 0.06);
}

[data-testid="stMetricValue"] {
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
    background: linear-gradient(135deg, #f0f0f5 0%, #a0a0b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

[data-testid="stMetricLabel"] {
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 0.7rem !important;
    color: var(--ob-text-tertiary) !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.5rem;
}

/* ─── 8. 结果状态卡片 ─── */
.result-high, .result-moderate, .result-low {
    border-radius: var(--ob-radius);
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.result-high {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.03));
    border: 1px solid rgba(16, 185, 129, 0.2);
    animation: glowPulse 3s ease-in-out infinite;
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 0 0 rgba(16, 185, 129, 0); }
    50% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 25px rgba(16, 185, 129, 0.08); }
}

.result-high::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981, #34d399, #10b981);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
}
.result-high:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 30px rgba(16, 185, 129, 0.12);
}

.result-moderate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(245, 158, 11, 0.03));
    border: 1px solid rgba(245, 158, 11, 0.2);
    animation: glowPulseAmber 3s ease-in-out infinite;
}
@keyframes glowPulseAmber {
    0%, 100% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 0 0 rgba(245, 158, 11, 0); }
    50% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 25px rgba(245, 158, 11, 0.08); }
}
.result-moderate::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.5);
}
.result-moderate:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 30px rgba(245, 158, 11, 0.12);
}

.result-low {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03));
    border: 1px solid rgba(239, 68, 68, 0.2);
    animation: glowPulseRed 3s ease-in-out infinite;
}
@keyframes glowPulseRed {
    0%, 100% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 0 0 rgba(239, 68, 68, 0); }
    50% { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15), 0 0 25px rgba(239, 68, 68, 0.08); }
}
.result-low::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ef4444, #f87171, #ef4444);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
}
.result-low:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 30px rgba(239, 68, 68, 0.12);
}

/* pKa 状态 */
.pka-acid, .pka-base, .pka-amphoteric {
    border-radius: var(--ob-radius);
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pka-acid {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(124, 58, 237, 0.03));
    border: 1px solid rgba(124, 58, 237, 0.2);
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15);
}
.pka-acid::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #a78bfa, #7c3aed);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.5);
}
.pka-acid:hover { transform: scale(1.02); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 25px rgba(124, 58, 237, 0.12); }

.pka-base {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(6, 182, 212, 0.03));
    border: 1px solid rgba(6, 182, 212, 0.2);
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15);
}
.pka-base::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #06b6d4, #22d3ee, #06b6d4);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
}
.pka-base:hover { transform: scale(1.02); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 25px rgba(6, 182, 212, 0.12); }

.pka-amphoteric {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(251, 191, 36, 0.03));
    border: 1px solid rgba(251, 191, 36, 0.2);
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.15);
}
.pka-amphoteric::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #fbbf24, #fde68a, #fbbf24);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
}
.pka-amphoteric:hover { transform: scale(1.02); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.2), 0 0 25px rgba(251, 191, 36, 0.12); }

/* ─── 9. Alert ─── */
.stAlert {
    border-radius: var(--ob-radius) !important;
    border: 1px solid !important;
    padding: 1.25rem !important;
    position: relative;
    overflow: hidden;
    animation: molecularDiffuse 0.25s ease both;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.stAlert::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}

.stAlert[data-baseweb="notification"][data-kind="positive"] {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.02)) !important;
    border-color: rgba(16, 185, 129, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="positive"]::before {
    background: linear-gradient(180deg, #10b981, #34d399);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.stAlert[data-baseweb="notification"][data-kind="info"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(124, 58, 237, 0.02)) !important;
    border-color: rgba(124, 58, 237, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="info"]::before {
    background: linear-gradient(180deg, #7c3aed, #a78bfa);
    box-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
}

.stAlert[data-baseweb="notification"][data-kind="warning"] {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(251, 191, 36, 0.02)) !important;
    border-color: rgba(251, 191, 36, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="warning"]::before {
    background: linear-gradient(180deg, #fbbf24, #fde68a);
    box-shadow: 0 0 8px rgba(251, 191, 36, 0.4);
}

.stAlert[data-baseweb="notification"][data-kind="negative"] {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.02)) !important;
    border-color: rgba(239, 68, 68, 0.18) !important;
}
.stAlert[data-baseweb="notification"][data-kind="negative"]::before {
    background: linear-gradient(180deg, #ef4444, #f87171);
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
}

.stAlert [data-testid="stMarkdownContainer"] {
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--ob-text-primary);
}

/* ─── 10. Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0 !important;
    padding: 0.5rem 1rem !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: var(--ob-text-secondary) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--ob-text-primary) !important;
    background: rgba(124, 58, 237, 0.05) !important;
    text-shadow: 0 0 12px rgba(124, 58, 237, 0.2);
}

.stTabs [aria-selected="true"] {
    color: var(--ob-nebula-light) !important;
    border-bottom-color: var(--ob-nebula) !important;
    background: linear-gradient(180deg, rgba(124, 58, 237, 0.1), transparent) !important;
    text-shadow: 0 0 16px rgba(124, 58, 237, 0.25);
    font-weight: 600 !important;
}

/* ─── 10.5 Radio（方式1 替代 selectbox）─── */
.stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 0 !important;
}

.stRadio > div > div {
    margin: 0 !important;
    padding: 0 !important;
}

.stRadio label {
    display: flex;
    align-items: center;
    padding: 0.5rem 0.875rem !important;
    margin: 0 !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease !important;
    font-size: 0.9rem !important;
    color: var(--ob-text-secondary) !important;
    background: transparent !important;
    border: none !important;
}

.stRadio label:hover {
    background: rgba(124, 58, 237, 0.12) !important;
    color: var(--ob-text-primary) !important;
}

/* 隐藏原生的 radio 圆圈 */
.stRadio label > span:first-child {
    display: none !important;
}

/* 选中的 radio 项 */
.stRadio [aria-checked="true"] + label,
.stRadio input:checked + div label,
.stRadio label[data-baseweb="radio"] [aria-checked="true"] {
    background: rgba(124, 58, 237, 0.18) !important;
    color: var(--ob-nebula-light) !important;
    font-weight: 600 !important;
}

/* radio 容器滚动 */
.stRadio > div {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 4px;
}

/* ─── 11. 媒体与图表 ─── */
.stImage > img {
    border-radius: var(--ob-radius);
    border: 1px solid var(--ob-border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stImage > img:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25), 0 0 30px rgba(124, 58, 237, 0.08);
}

.js-plotly-plot, .stPlotlyChart {
    border-radius: var(--ob-radius);
    overflow: hidden;
}

/* ─── 12. 分隔线与页脚 ─── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(124, 58, 237, 0.2) 20%, rgba(6, 182, 212, 0.15) 50%, rgba(124, 58, 237, 0.2) 80%, transparent 100%);
    margin: 2rem 0;
    position: relative;
}
hr::after {
    content: '';
    position: absolute;
    left: 50%; top: -2px;
    transform: translateX(-50%);
    width: 6px; height: 6px;
    background: rgba(124, 58, 237, 0.4);
    border-radius: 50%;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.4);
}

.footer {
    text-align: center;
    padding: 2.5rem 1.5rem;
    color: var(--ob-text-tertiary);
    font-size: 0.8125rem;
    line-height: 1.6;
    border-top: 1px solid var(--ob-border);
    margin-top: 3rem;
    position: relative;
    font-family: 'Cascadia Code', 'Consolas', monospace;
}
.footer::before {
    content: '';
    position: absolute;
    top: -1px; left: 50%;
    transform: translateX(-50%);
    width: 120px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.4), transparent);
}

/* ─── 13. 加载状态 ─── */
.stSpinner > div {
    border-color: var(--ob-nebula) transparent transparent transparent !important;
}

.loading-bar {
    height: 2px;
    background: rgba(124, 58, 237, 0.1);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.loading-bar::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 25%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--ob-nebula), transparent);
    animation: electronJump 1.5s ease-in-out infinite;
}

/* ─── 14. 辅助工具类 ─── */
.text-gradient {
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-primary {
    background: rgba(124, 58, 237, 0.12);
    color: #c4b5fd;
    border: 1px solid rgba(124, 58, 237, 0.15);
}

.badge-success {
    background: rgba(16, 185, 129, 0.12);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.15);
}

.badge-warn {
    background: rgba(251, 191, 36, 0.12);
    color: #fcd34d;
    border: 1px solid rgba(251, 191, 36, 0.15);
}

/* ─── 15. 滚动条 ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(124, 58, 237, 0.3), rgba(6, 182, 212, 0.25));
    border-radius: 10px;
    border: 1px solid transparent;
    background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, rgba(124, 58, 237, 0.5), rgba(6, 182, 212, 0.4));
    background-clip: content-box;
}

/* 文字选中效果 */
::selection {
    background: rgba(124, 58, 237, 0.3);
    color: var(--ob-text-primary);
}

/* Streamlit 原生组件适配 */
.stMarkdown, .stText, p, li, span { color: var(--ob-text-primary); }
.stCaption {
    color: var(--ob-text-tertiary) !important;
    font-size: 0.8125rem !important;
    font-family: 'Cascadia Code', 'Consolas', monospace !important;
}

/* st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(155deg, rgba(30, 30, 46, 0.65) 0%, rgba(15, 15, 25, 0.45) 100%) !important;
    border: 1px solid var(--ob-border) !important;
    border-radius: var(--ob-radius) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15), 0 4px 12px -2px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(124, 58, 237, 0.12) !important;
    box-shadow: 0 4px 12px -2px rgba(0,0,0,0.25), 0 0 20px rgba(124, 58, 237, 0.05), inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: linear-gradient(155deg, rgba(30, 30, 46, 0.6) 0%, rgba(15, 15, 25, 0.4) 100%) !important;
    border: 1px solid var(--ob-border) !important;
    border-radius: var(--ob-radius) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12) !important;
}

[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.18), 0 0 0 1px rgba(124, 58, 237, 0.06) !important;
}

/* 隐藏 Streamlit 默认顶栏 */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* 响应式 */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .gradient-title { font-size: 2rem; }
    .card-container { padding: 1.25rem; }
}

/* ─── 图表玻璃容器 ─── */
.chart-glass-wrapper {
    border-radius: var(--ob-radius);
    overflow: hidden;
    background: linear-gradient(155deg, rgba(30, 30, 46, 0.50) 0%, rgba(15, 15, 25, 0.35) 100%);
    backdrop-filter: blur(8px) saturate(110%);
    -webkit-backdrop-filter: blur(8px) saturate(110%);
    border: 1px solid var(--ob-border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.03);
    padding: 0.5rem;
    margin-bottom: 1rem;
}

/* ─── Tab 内容区玻璃底座 ─── */
.tab-glass-panel {
    background: linear-gradient(180deg, rgba(20, 20, 35, 0.40) 0%, rgba(13, 13, 20, 0.25) 100%);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 0 0 var(--ob-radius) var(--ob-radius);
    padding: 1.5rem;
    margin-top: -0.5rem;
    animation: molecularDiffuse 0.3s ease both;
}

/* ─── 骨架屏动画 ─── */
@keyframes skeletonShimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.skeleton-box {
    background: linear-gradient(90deg, rgba(30,30,46,0.6) 25%, rgba(50,50,70,0.5) 50%, rgba(30,30,46,0.6) 75%);
    background-size: 200% 100%;
    border-radius: var(--ob-radius-sm);
    animation: skeletonShimmer 1.8s ease infinite;
    min-height: 120px;
    margin-bottom: 1rem;
}

/* ─── Tab 标签增强 ─── */
.stTabs [data-baseweb="tab-panel"] {
    animation: molecularDiffuse 0.3s ease both;
}

/* ─── prefers-reduced-motion ─── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .stApp::before, .stApp::after { display: none; }
    .cursor-glow { display: none; }
    .skeleton-box { animation: none; background: rgba(30,30,46,0.6); }
}

/* ─── 文字选中增强 ─── */
::selection {
    background: rgba(124, 58, 237, 0.35);
    color: #ffffff;
    text-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
}

/* ─── 响应式图表容器 ─── */
@media (max-width: 768px) {
    .chart-glass-wrapper { padding: 0.25rem; border-radius: var(--ob-radius-sm); }
    .tab-glass-panel { padding: 1rem; }
}

/* ─── 交互增强：鼠标悬停光效 ─── */
.cursor-glow {
    position: fixed;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.22) 0%, rgba(124, 58, 237, 0.08) 40%, transparent 70%);
    pointer-events: none;
    z-index: 9999;
    transform: translate3d(-50%, -50%, 0);
    transition: opacity 0.3s ease;
    will-change: transform;
}
</style>"""


def inject_theme_css():
    """Inject the deep space theme CSS into the Streamlit page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
