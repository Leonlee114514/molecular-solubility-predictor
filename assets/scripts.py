"""
DisSolve - Client-side JavaScript injection.
Contains dropdown override, starfield canvas, mouse glow, and glossary scripts.
"""

import streamlit as st


def _render_html(html_content, height=1):
    """Render HTML/JS via st.components.v1.html (same-origin, window.parent access works)."""
    st.components.v1.html(html_content, height=height)


_DROPDOWN_OVERRIDE_JS = """<script>

(function() {
    'use strict';
    var win = window.parent || window;
    var doc = win.document;
    if (!doc || !doc.body) return;

    var DARK_BG = '#1a1a2e';
    var TEXT = '#e0e0e0';
    var HOVER_BG = 'rgba(124, 58, 237, 0.25)';
    var SELECTED_BG = 'rgba(124, 58, 237, 0.35)';
    var SELECTED_COLOR = '#ffffff';
    var fixedMenus = new WeakSet();

    function applyMenuStyle(menu) {
        if (fixedMenus.has(menu)) return;
        fixedMenus.add(menu);
        menu.style.background = DARK_BG;
        menu.style.backgroundColor = DARK_BG;
        menu.style.border = '1px solid rgba(255, 255, 255, 0.08)';
        menu.style.borderRadius = '12px';
        menu.style.boxShadow = '0 10px 30px -5px rgba(0,0,0,0.4)';
        menu.style.color = TEXT;
    }

    function fixAllMenus() {
        var menus = doc.querySelectorAll('[data-baseweb="menu"], [data-baseweb="popover"], [role="listbox"]');
        menus.forEach(function(menu) {
            applyMenuStyle(menu);
            // clean inner divs
            var divs = menu.querySelectorAll('div');
            for (var i = 0; i < divs.length; i++) {
                divs[i].style.background = 'transparent';
                divs[i].style.backgroundColor = 'transparent';
                divs[i].style.backgroundImage = 'none';
            }
            // fix option rows
            var opts = menu.querySelectorAll('[role="option"]');
            opts.forEach(function(opt) {
                opt.style.color = TEXT;
                opt.style.background = 'transparent';
                opt.style.backgroundColor = 'transparent';
                if (opt.getAttribute('aria-selected') === 'true') {
                    opt.style.background = SELECTED_BG;
                    opt.style.backgroundColor = SELECTED_BG;
                    opt.style.color = SELECTED_COLOR;
                    opt.style.fontWeight = '600';
                }
                if (!opt.dataset.obHover) {
                    opt.dataset.obHover = '1';
                    opt.addEventListener('mouseenter', function() {
                        if (opt.getAttribute('aria-selected') !== 'true') {
                            opt.style.background = HOVER_BG;
                            opt.style.backgroundColor = HOVER_BG;
                            opt.style.color = '#ffffff';
                        }
                    });
                    opt.addEventListener('mouseleave', function() {
                        if (opt.getAttribute('aria-selected') !== 'true') {
                            opt.style.background = 'transparent';
                            opt.style.backgroundColor = 'transparent';
                            opt.style.color = TEXT;
                        }
                    });
                }
            });
        });
    }

    var pending = false;
    function scheduleFix() {
        if (pending) return;
        pending = true;
        requestAnimationFrame(function() {
            pending = false;
            fixAllMenus();
        });
    }

    var observer = new MutationObserver(function(mutations) {
        for (var m = 0; m < mutations.length; m++) {
            var added = mutations[m].addedNodes;
            for (var n = 0; n < added.length; n++) {
                if (added[n].nodeType === 1) {
                    scheduleFix();
                    break;
                }
            }
        }
    });
    observer.observe(doc.documentElement || doc.body, { childList: true, subtree: true });
    doc.body.addEventListener('click', function() { setTimeout(fixAllMenus, 100); });
    fixAllMenus();
})();
</script>"""

_PARTICLE_STARRY_BG_JS = """<script>
(function() {
    'use strict';
    console.log('[DisSolve] Starfield script loaded');
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        console.log('[DisSolve] prefers-reduced-motion is enabled, skipping starfield');
        return;
    }
    var win = window.parent || window;
    var doc = win.document;
    var canvas = doc.getElementById('ob-starfield');
    if (!canvas) {
        canvas = doc.createElement('canvas');
        canvas.id = 'ob-starfield';
        canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:1;pointer-events:none;will-change:transform;';
        var app = doc.querySelector('.stApp');
        if (app) { app.insertBefore(canvas, app.firstChild); }
        else { doc.body.insertBefore(canvas, doc.body.firstChild); }
    }
    var ctx = canvas.getContext('2d');
    var W, H;
    function resize() { W = canvas.width = (win.innerWidth || 1920); H = canvas.height = (win.innerHeight || 1080); }
    resize();
    win.addEventListener('resize', resize);
    if (!doc.__obStarMouseBound) {
        doc.__obStarMouseBound = true;
        doc.addEventListener('mousemove', function(e) { mouseStarX = e.clientX; mouseStarY = e.clientY; mouseTrail.push({x: e.clientX, y: e.clientY, age: 0}); if (mouseTrail.length > maxTrail) mouseTrail.shift(); }, { passive: true });
        doc.addEventListener('click', function(e) { bursts.push({x: e.clientX, y: e.clientY, radius: 0, maxRadius: 200, life: 1.0}); }, { passive: true });
    }
    var layers = [
        { count: 50, minR: 1.8, maxR: 3.5, speed: 0.08, colors: ['rgba(196,181,253,', 'rgba(103,232,249,', 'rgba(233,213,255,', 'rgba(255,255,255,'] },
        { count: 70, minR: 1.0, maxR: 2.2, speed: 0.04, colors: ['rgba(167,139,250,', 'rgba(34,211,238,', 'rgba(139,92,246,', 'rgba(253,224,71,'] },
        { count: 80, minR: 0.5, maxR: 1.2, speed: 0.02, colors: ['rgba(196,181,253,', 'rgba(103,232,249,', 'rgba(255,255,255,'] }
    ];
    var stars = [];
    layers.forEach(function(layer, li) {
        for (var i = 0; i < layer.count; i++) {
            var colorBase = layer.colors[Math.floor(Math.random() * layer.colors.length)];
            stars.push({ x: Math.random() * W, y: Math.random() * H, r: layer.minR + Math.random() * (layer.maxR - layer.minR), baseR: layer.minR + Math.random() * (layer.maxR - layer.minR), vx: (Math.random() - 0.5) * layer.speed, vy: (Math.random() - 0.5) * layer.speed, color: colorBase, alpha: 0.5 + Math.random() * 0.5, twinkleSpeed: 0.005 + Math.random() * 0.015, twinklePhase: Math.random() * Math.PI * 2, layer: li });
        }
    });
    var mouseStarX = W / 2, mouseStarY = H / 2;
    var mouseTrail = [], maxTrail = 20, bursts = [];
    var frame = 0;
    function animate() {
        var bgGrad = ctx.createLinearGradient(0, 0, 0, H);
        bgGrad.addColorStop(0, '#0f0f1c'); bgGrad.addColorStop(0.3, '#131328'); bgGrad.addColorStop(0.5, '#181830'); bgGrad.addColorStop(0.7, '#131328'); bgGrad.addColorStop(1, '#0f0f1c');
        ctx.fillStyle = bgGrad; ctx.clearRect(0, 0, W, H);
        var nebulas = [
            {x: W*0.18, y: H*0.25, rx: W*0.35, ry: H*0.28, color: 'rgba(124,58,237,'},
            {x: W*0.82, y: H*0.12, rx: W*0.28, ry: H*0.22, color: 'rgba(6,182,212,'},
            {x: W*0.50, y: H*0.80, rx: W*0.25, ry: H*0.20, color: 'rgba(251,191,36,'},
            {x: W*0.50, y: H*0.40, rx: W*0.40, ry: H*0.32, color: 'rgba(67,56,202,'}
        ];
        nebulas.forEach(function(n) {
            var g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, Math.max(n.rx, n.ry));
            g.addColorStop(0, n.color + '0.25)'); g.addColorStop(0.5, n.color + '0.08)'); g.addColorStop(1, n.color + '0)');
            ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(n.x, n.y, n.rx, n.ry, 0, 0, Math.PI*2); ctx.fill();
        });
        frame++;
        var lineCount = 0, maxLines = 30;
        for (var i = 0; i < stars.length && lineCount < maxLines; i++) {
            var dmx = stars[i].x - mouseStarX, dmy = stars[i].y - mouseStarY;
            var dMouse = Math.sqrt(dmx*dmx + dmy*dmy);
            if (dMouse > 200) continue;
            for (var j = i + 1; j < stars.length && lineCount < maxLines; j++) {
                if (stars[j].layer > 1) continue;
                var dx = stars[i].x - stars[j].x, dy = stars[i].y - stars[j].y;
                var dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    var lineAlpha = 0.1 * (1 - dist/120) * (1 - dMouse/200);
                    ctx.beginPath(); ctx.moveTo(stars[i].x, stars[i].y); ctx.lineTo(stars[j].x, stars[j].y);
                    ctx.strokeStyle = 'rgba(167,139,250,' + lineAlpha + ')'; ctx.lineWidth = 0.6; ctx.stroke();
                    lineCount++;
                }
            }
        }
        for (var b = bursts.length - 1; b >= 0; b--) { bursts[b].radius += 5; bursts[b].life = Math.max(0, 1 - bursts[b].radius / bursts[b].maxRadius); if (bursts[b].life <= 0) bursts.splice(b, 1); }
        for (var i = 0; i < stars.length; i++) {
            var s = stars[i];
            for (var b = 0; b < bursts.length; b++) {
                var bdx = s.x - bursts[b].x, bdy = s.y - bursts[b].y;
                var bDist = Math.sqrt(bdx*bdx + bdy*bdy);
                if (bDist < bursts[b].radius && bDist < bursts[b].maxRadius && bDist > 2) {
                    var force = (1 - bDist / bursts[b].maxRadius) * bursts[b].life * 2.5 * (1 - s.layer * 0.3);
                    s.vx += (bdx / bDist) * force; s.vy += (bdy / bDist) * force;
                }
            }
            s.x += s.vx; s.y += s.vy;
            s.vx *= 0.992; s.vy *= 0.992;
            var mdx = mouseStarX - s.x, mdy = mouseStarY - s.y;
            var mDist = Math.sqrt(mdx*mdx + mdy*mdy);
            if (mDist < 250 && mDist > 5) { var attract = (250 - mDist) / 250 * 0.35 * (1 - s.layer * 0.3); s.x += (mdx / mDist) * attract; s.y += (mdy / mDist) * attract; }
            if (s.x < -20) s.x = W + 20; if (s.x > W + 20) s.x = -20;
            if (s.y < -20) s.y = H + 20; if (s.y > H + 20) s.y = -20;
            var twinkle = Math.sin(frame * s.twinkleSpeed + s.twinklePhase);
            var curAlpha = s.alpha * (0.5 + 0.5 * twinkle);
            s.r = s.baseR * (0.8 + 0.2 * twinkle);
            if (mDist < 200) { curAlpha = Math.min(1, curAlpha + (200 - mDist)/200 * 0.4); s.r *= 1 + (200 - mDist)/200 * 0.5; }
            var glowR = s.r * (s.layer === 0 ? 8 : 5);
            var glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, glowR);
            glow.addColorStop(0, s.color + (curAlpha * 0.5) + ')'); glow.addColorStop(1, s.color + '0)');
            ctx.beginPath(); ctx.arc(s.x, s.y, glowR, 0, Math.PI*2); ctx.fillStyle = glow; ctx.fill();
            ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI*2); ctx.fillStyle = s.color + curAlpha + ')'; ctx.fill();
        }
        for (var t = 0; t < mouseTrail.length; t++) {
            var pt = mouseTrail[t]; pt.age++; var life = 1 - pt.age / 40; if (life <= 0) continue;
            var tr = 2 + life * 4;
            var tg = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, tr);
            tg.addColorStop(0, 'rgba(196,181,253,' + (life * 0.35) + ')'); tg.addColorStop(1, 'rgba(124,58,237,0)');
            ctx.beginPath(); ctx.arc(pt.x, pt.y, tr, 0, Math.PI*2); ctx.fillStyle = tg; ctx.fill();
        }
        while (mouseTrail.length > 0 && mouseTrail[0].age > 40) mouseTrail.shift();
        var mGlow = ctx.createRadialGradient(mouseStarX, mouseStarY, 0, mouseStarX, mouseStarY, 60);
        mGlow.addColorStop(0, 'rgba(196,181,253,0.15)'); mGlow.addColorStop(0.4, 'rgba(124,58,237,0.06)'); mGlow.addColorStop(1, 'rgba(124,58,237,0)');
        ctx.beginPath(); ctx.arc(mouseStarX, mouseStarY, 60, 0, Math.PI*2); ctx.fillStyle = mGlow; ctx.fill();
        for (var b = 0; b < bursts.length; b++) {
            var alpha = bursts[b].life * 0.4;
            ctx.beginPath(); ctx.arc(bursts[b].x, bursts[b].y, bursts[b].radius, 0, Math.PI*2);
            ctx.strokeStyle = 'rgba(167,139,250,' + alpha + ')'; ctx.lineWidth = 2 * bursts[b].life; ctx.stroke();
            var bGlow = ctx.createRadialGradient(bursts[b].x, bursts[b].y, 0, bursts[b].x, bursts[b].y, bursts[b].radius);
            bGlow.addColorStop(0, 'rgba(196,181,253,' + (alpha * 0.6) + ')'); bGlow.addColorStop(0.4, 'rgba(124,58,237,' + (alpha * 0.2) + ')'); bGlow.addColorStop(1, 'rgba(124,58,237,0)');
            ctx.beginPath(); ctx.arc(bursts[b].x, bursts[b].y, bursts[b].radius, 0, Math.PI*2); ctx.fillStyle = bGlow; ctx.fill();
        }
        animFrameId = requestAnimationFrame(animate);
    }
    var animFrameId = requestAnimationFrame(animate);
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) { if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; } }
        else { if (!animFrameId) animFrameId = requestAnimationFrame(animate); }
    });
})();
</script>"""

_MOUSE_GLOW_TILT_JS = """<script>

(function() {
    'use strict';
    let glow = window.parent.document.querySelector('.cursor-glow');
    if (!glow) {
        glow = window.parent.document.createElement('div');
        glow.className = 'cursor-glow';
        window.parent.document.body.appendChild(glow);
    }

    let win2 = window.parent || window;
    let doc2 = win2.document;
    let mouseX = win2.innerWidth / 2, mouseY = win2.innerHeight / 2;
    let currentX = mouseX, currentY = mouseY;
    let moveTimeout;

    console.log('[DisSolve] Cursor glow script loaded, win size: ' + win2.innerWidth + 'x' + win2.innerHeight);

    if (!doc2.__obGlowMouseBound) {
        doc2.__obGlowMouseBound = true;
        doc2.addEventListener('mousemove', function(e) {
            mouseX = e.clientX; mouseY = e.clientY;
            glow.style.opacity = '1';
            clearTimeout(moveTimeout);
            moveTimeout = setTimeout(() => { glow.style.opacity = '0'; }, 150);
        }, { passive: true });
    }
    if (!doc2.__obCursorLeaveBound) {
        doc2.__obCursorLeaveBound = true;
        doc2.addEventListener('mouseleave', () => glow.style.opacity = '0');
    }

    (function animate() {
        currentX += (mouseX - currentX) * 0.08;
        currentY += (mouseY - currentY) * 0.08;
        glow.style.transform = 'translate3d(' + (currentX - 150) + 'px, ' + (currentY - 150) + 'px, 0)';
        requestAnimationFrame(animate);
    })();
    console.log('[DisSolve] Cursor glow initialized at ' + currentX + ',' + currentY);

    // 卡片 3D tilt 效果（更克制的角度，避免与CSS hover冲突）
    function bindTilt() {
        const cards = window.parent.document.querySelectorAll('.card-container:not([data-tilt-bound]), [data-testid="stVerticalBlockBorderWrapper"]:not([data-tilt-bound])');
        cards.forEach(card => {
            card.dataset.tiltBound = '1';
            card.addEventListener('mousemove', function(e) {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = ((y - centerY) / centerY * -1.5).toFixed(2);
                const rotateY = ((x - centerX) / centerX * 1.5).toFixed(2);
                card.style.transform = 'perspective(1000px) rotateX(' + rotateX + 'deg) rotateY(' + rotateY + 'deg) translateY(-4px)';
            }, { passive: true });
            card.addEventListener('mouseleave', function() {
                card.style.transform = '';
            });
        });
    }
    bindTilt();
    setTimeout(bindTilt, 1500); // Streamlit动态渲染后重绑定

    // 按钮点击涟漪效果（品牌紫色 + 数量限制3个）
    function bindRipple() {
        const buttons = window.parent.document.querySelectorAll('.stButton > button:not([data-ripple-bound])');
        buttons.forEach(btn => {
            btn.dataset.rippleBound = '1';
            btn.style.position = 'relative';
            btn.style.overflow = 'hidden';
            btn.addEventListener('click', function(e) {
                const existing = btn.querySelectorAll('.ob-ripple');
                if (existing.length >= 3) existing[0].remove();
                const ripple = window.parent.document.createElement('span');
                ripple.className = 'ob-ripple';
                ripple.style.cssText = `
                    position: absolute; border-radius: 50%;
                    background: radial-gradient(circle, rgba(167,139,250,0.35) 0%, rgba(124,58,237,0.15) 60%, transparent 100%);
                    transform: scale(0); animation: rippleAnim 0.7s ease-out forwards;
                    pointer-events: none;
                `;
                const rect = btn.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height) * 1.4;
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = (e.clientX - rect.left - size/2) + 'px';
                ripple.style.top = (e.clientY - rect.top - size/2) + 'px';
                btn.appendChild(ripple);
                setTimeout(() => ripple.remove(), 700);
            });
        });
    }
    bindRipple();
    setTimeout(bindRipple, 1500);

    // 动态添加涟漪动画（仅添加一次）
    if (!window.parent.document.getElementById('ob-ripple-style')) {
        const style = window.parent.document.createElement('style');
        style.id = 'ob-ripple-style';
        style.textContent = '@keyframes rippleAnim { 0% { transform: scale(0); opacity: 1; } 100% { transform: scale(4); opacity: 0; } }';
        window.parent.document.head.appendChild(style);
    }
})();
</script>"""

_GLOSSARY_JS = """<script>

(function() {
    'use strict';
    var win = window.parent || window;
    var doc = win.document;

    // ═══════════════════════════════════════════
    // 注入样式到父文档
    // ═══════════════════════════════════════════
    if (!doc.getElementById('gloss-style')) {
        var style = doc.createElement('style');
        style.id = 'gloss-style';
        style.textContent = ''
            + '.gloss-term {'
            + '  border-bottom: 1.5px dashed rgba(124, 58, 237, 0.55);'
            + '  cursor: pointer;'
            + '  color: #c4b5fd;'
            + '  transition: color 0.15s, border-color 0.15s;'
            + '  position: relative;'
            + '}'
            + '.gloss-term:hover {'
            + '  color: #e0d4ff;'
            + '  border-bottom-color: rgba(167, 139, 250, 0.8);'
            + '}'
            + '#gloss-popup {'
            + '  position: fixed;'
            + '  z-index: 99999;'
            + '  max-width: 420px;'
            + '  background: linear-gradient(155deg, rgba(30, 30, 50, 0.96) 0%, rgba(18, 18, 35, 0.94) 100%);'
            + '  backdrop-filter: blur(14px) saturate(130%);'
            + '  -webkit-backdrop-filter: blur(14px) saturate(130%);'
            + '  border: 1px solid rgba(124, 58, 237, 0.25);'
            + '  border-radius: 16px;'
            + '  padding: 1.25rem 1.5rem;'
            + '  box-shadow: 0 20px 50px -12px rgba(0,0,0,0.55), 0 0 0 1px rgba(124, 58, 237, 0.08), 0 0 30px rgba(124, 58, 237, 0.1);'
            + '  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;'
            + '  pointer-events: auto;'
            + '  opacity: 0;'
            + '  transform: translateY(6px) scale(0.97);'
            + '  transition: opacity 0.18s ease, transform 0.18s ease;'
            + '  display: none;'
            + '}'
            + '#gloss-popup.show {'
            + '  display: block;'
            + '  opacity: 1;'
            + '  transform: translateY(0) scale(1);'
            + '}'
            + '#gloss-popup .gloss-en {'
            + '  font-size: 1.15rem;'
            + '  font-weight: 700;'
            + '  color: #f0f0f5;'
            + '  margin-bottom: 0.2rem;'
            + '  letter-spacing: -0.01em;'
            + '}'
            + '#gloss-popup .gloss-cn {'
            + '  font-size: 0.9rem;'
            + '  color: #a78bfa;'
            + '  font-weight: 600;'
            + '  margin-bottom: 0.6rem;'
            + '}'
            + '#gloss-popup .gloss-def {'
            + '  font-size: 0.85rem;'
            + '  color: #c0c0d0;'
            + '  line-height: 1.65;'
            + '}'
            + '#gloss-popup .gloss-def-en {'
            + '  font-size: 0.8rem;'
            + '  color: #8b8b9b;'
            + '  line-height: 1.55;'
            + '  margin-top: 0.45rem;'
            + '  padding-top: 0.45rem;'
            + '  border-top: 1px solid rgba(255,255,255,0.06);'
            + '}'
            + '#gloss-overlay {'
            + '  position: fixed;'
            + '  inset: 0;'
            + '  z-index: 99998;'
            + '  pointer-events: auto;'
            + '  display: none;'
            + '}'
            + '#gloss-overlay.show { display: block; }';
        doc.head.appendChild(style);
    }

    // ═══════════════════════════════════════════
    // 化学术语词汇表（中英双语 — 每个概念含中英文双 key）
    // ═══════════════════════════════════════════
    var GLOSS_ENTRIES = [
        { keys: ["溶解度", "Solubility"], en: "Solubility", cn: "溶解度", def: "物质在特定温度下溶解于溶剂中形成均匀溶液的最大量。", defEn: "The maximum amount of a substance that can dissolve in a solvent at a given temperature to form a homogeneous solution." },
        { keys: ["水溶性", "Aqueous Solubility"], en: "Aqueous Solubility", cn: "水溶性", def: "物质在水中溶解的能力，通常用 logS 表示。水溶性由分子的极性、氢键能力和疏水性共同决定。", defEn: "The ability of a substance to dissolve in water, typically expressed as logS. Determined by polarity, hydrogen-bonding capacity, and hydrophobicity." },
        { keys: ["logS"], en: "logS", cn: "溶解度对数", def: "水溶解度的常用对数值。logS > 0 表示易溶，-2 < logS < 0 表示中等溶解，logS < -2 表示难溶。", defEn: "The base-10 logarithm of aqueous solubility. logS > 0 = highly soluble; -2 < logS < 0 = moderately soluble; logS < -2 = poorly soluble." },
        { keys: ["pKa"], en: "pKa", cn: "酸解离常数", def: "酸解离常数的负对数，衡量分子给出质子的倾向。pKa 越低酸性越强，pKa 越高碱性越强。pKa < 5 为酸性，5-9 为两性/中性，> 9 为碱性。", defEn: "The negative logarithm of the acid dissociation constant. Lower pKa = stronger acid; higher pKa = stronger base. pKa < 5 = acidic, 5-9 = amphoteric/neutral, > 9 = basic." },
        { keys: ["SMILES"], en: "SMILES", cn: "简化分子线性输入规范", def: "Simplified Molecular Input Line Entry System — 用 ASCII 字符串描述分子结构的标准表示法。原子由元素符号表示，键通过隐含规则推导。", defEn: "Simplified Molecular Input Line Entry System — a notation that encodes molecular structures as ASCII strings. Atoms are represented by element symbols; bonds are inferred through implicit rules." },
        { keys: ["分子量", "Molecular Weight", "MolWt", "Mol Wt"], en: "Molecular Weight (MolWt)", cn: "分子量", def: "分子中所有原子质量的总和，单位为 g/mol。影响分子的扩散速率、膜渗透性和溶解行为。", defEn: "The sum of atomic masses in a molecule (g/mol). Influences diffusion rate, membrane permeability, and dissolution behavior." },
        { keys: ["LogP", "Partition Coefficient"], en: "LogP (Partition Coefficient)", cn: "脂水分配系数", def: "分子在正辛醇（油相）和水相之间分配比的对数值。LogP 越高脂溶性越强，LogP 越低水溶性越好。用于评估药物的吸收、分布和毒性。", defEn: "The logarithm of a molecule's partition ratio between n-octanol (oil) and water. Higher LogP = more lipophilic; lower LogP = more hydrophilic. Used to predict drug absorption, distribution, and toxicity." },
        { keys: ["TPSA", "Topological Polar Surface Area", "极性表面积"], en: "Topological Polar Surface Area (TPSA)", cn: "拓扑极性表面积", def: "分子中极性原子（氧、氮及与其相连的氢）所占据的表面积总和（单位 Å²）。TPSA 越高通常意味着水溶性越好，也是预测药物口服吸收的重要指标。", defEn: "The sum of surface areas of polar atoms (oxygen, nitrogen, and attached hydrogens) in a molecule (Å²). Higher TPSA generally means better water solubility. A key predictor of oral drug absorption." },
        { keys: ["氢键供体", "H-Bond Donors", "H-Bond Donor", "HBD", "Hydrogen Bond Donor"], en: "Hydrogen Bond Donor (HBD)", cn: "氢键供体", def: "分子中能提供氢原子形成氢键的基团，如 -OH（羟基）和 -NH（氨基）。氢键供体数量越多，分子与水形成氢键的能力越强，水溶性通常越好。", defEn: "Groups that can donate a hydrogen atom to form a hydrogen bond, e.g. -OH (hydroxyl) and -NH (amino). More HBDs generally mean stronger hydrogen bonding with water and better solubility." },
        { keys: ["氢键受体", "H-Bond Acceptors", "H-Bond Acceptor", "HBA", "Hydrogen Bond Acceptor", "H-Acceptors"], en: "Hydrogen Bond Acceptor (HBA)", cn: "氢键受体", def: "分子中含有孤对电子、能接受氢原子的原子，如 O、N、F。氢键受体越多，分子越容易与水分子形成氢键网络。", defEn: "Atoms with lone-pair electrons that can accept hydrogen atoms, e.g. O, N, F. More HBAs allow the molecule to form more hydrogen bonds with water." },
        { keys: ["可旋转键", "Rotatable Bonds", "Rotatable Bond"], en: "Rotatable Bonds", cn: "可旋转键", def: "分子中能够自由旋转的单键数量。可旋转键越多，分子柔性越大，可能影响其与靶点结合的能力和结晶性。", defEn: "The number of single bonds that can freely rotate. More rotatable bonds = greater molecular flexibility, which can affect target binding and crystallinity." },
        { keys: ["芳香环", "Aromatic Rings", "Aromatic Ring"], en: "Aromatic Rings", cn: "芳香环", def: "具有共轭 π 电子体系的平面环状结构。芳香环增强分子刚性，提供 π-π 堆积作用，常见于药物分子中用于与靶点结合。", defEn: "Planar ring structures with conjugated π-electron systems. Aromatic rings increase molecular rigidity and provide π-π stacking interactions, commonly found in drug molecules for target binding." },
        { keys: ["芳香性", "Aromaticity"], en: "Aromaticity", cn: "芳香性", def: "具有共轭 π 电子体系的环状分子所表现出的特殊稳定性。根据 Hückel 规则，具有 4n+2 个 π 电子的平面环状分子具有芳香性。芳香性使分子更稳定，影响其反应活性、pKa 和与靶点的 π-π 堆积作用。", defEn: "The special stability exhibited by cyclic molecules with conjugated π-electron systems. According to Hückel's rule, planar cyclic molecules with 4n+2 π electrons are aromatic. Aromaticity stabilizes molecules and affects reactivity, pKa, and π-π stacking interactions with targets." },
        { keys: ["脂肪环", "Aliphatic Rings", "Aliphatic Ring"], en: "Aliphatic Rings", cn: "脂肪环", def: "不含芳香性的碳环结构，如环己烷。脂肪环比芳香环更柔性，影响分子的三维构象和溶解性。", defEn: "Non-aromatic carbon ring structures, e.g. cyclohexane. More flexible than aromatic rings, influencing 3D conformation and solubility." },
        { keys: ["摩根指纹", "Morgan Fingerprint", "Morgan FP", "ECFP", "Morgan Fingerprint (ECFP)"], en: "Morgan Fingerprint (ECFP)", cn: "摩根分子指纹", def: "一种将分子结构编码为固定长度二进制向量的方法。每个比特位表示分子中是否存在特定的子结构片段。radius=2 表示考虑每个原子周围 2 个键范围内的环境。", defEn: "A method that encodes molecular structure as a fixed-length binary vector. Each bit indicates the presence of a specific substructure fragment. radius=2 considers the environment within 2 bonds of each atom." },
        { keys: ["随机森林", "Random Forest"], en: "Random Forest", cn: "随机森林", def: "一种集成学习算法，通过构建多棵决策树并取平均来进行回归预测。每棵树在随机数据子集和随机特征子集上训练，抗过拟合能力强。本应用使用 200 棵树。", defEn: "An ensemble learning algorithm that builds multiple decision trees and averages their predictions. Each tree is trained on a random subset of data and features. Robust against overfitting. This app uses 200 trees." },
        { keys: ["SHAP"], en: "SHAP (SHapley Additive exPlanations)", cn: "SHAP 可解释性分析", def: "基于博弈论中 Shapley 值的模型解释方法。SHAP 值量化每个特征对预测结果的贡献大小和方向。正值推动预测值升高（更易溶），负值推动预测值降低（更难溶）。", defEn: "A model explanation method based on Shapley values from game theory. SHAP values quantify each feature's contribution magnitude and direction to the prediction. Positive values push the prediction higher (more soluble), negative values push it lower (less soluble)." },
        { keys: ["疏水性", "Hydrophobicity"], en: "Hydrophobicity", cn: "疏水性", def: "分子排斥水的倾向。疏水基团（如长烷基链、芳香环）倾向于聚集在一起以减少与水接触，这是药物设计中需要考虑的重要因素。", defEn: "The tendency of a molecule to repel water. Hydrophobic groups (e.g. long alkyl chains, aromatic rings) tend to cluster together to minimize water contact — a key consideration in drug design." },
        { keys: ["亲水性", "Hydrophilicity"], en: "Hydrophilicity", cn: "亲水性", def: "分子吸引水或与水相互作用的倾向。亲水基团（如 -OH、-COOH、-NH2）能通过氢键或离子-偶极作用与水分子结合，促进溶解。", defEn: "The tendency of a molecule to attract or interact with water. Hydrophilic groups (e.g. -OH, -COOH, -NH2) bind water via hydrogen bonds or ion-dipole interactions, promoting dissolution." },
        { keys: ["电离", "Ionization"], en: "Ionization", cn: "电离", def: "分子在溶液中失去或获得质子 (H+) 形成带电离子的过程。电离状态随环境 pH 变化，直接影响药物的吸收、分布和排泄。分子态脂溶性高易吸收，离子态水溶性好易排泄。", defEn: "The process by which a molecule loses or gains a proton (H+) in solution, forming charged ions. The ionization state depends on pH and directly affects drug absorption, distribution, and excretion. Unionized forms are lipid-soluble and easily absorbed; ionized forms are water-soluble and readily excreted." },
        { keys: ["分子态", "Unionized Form", "非电离形式"], en: "Unionized Form", cn: "分子态（非电离形式）", def: "分子在中性状态下未发生电离的形式。分子态通常脂溶性较高，更容易穿透细胞膜被吸收。", defEn: "The neutral, non-ionized form of a molecule. Unionized forms are typically more lipid-soluble and can more easily cross cell membranes for absorption." },
        { keys: ["生物利用度", "Bioavailability"], en: "Bioavailability", cn: "生物利用度", def: "药物进入体循环后到达作用部位的百分率。受溶解度、pKa、分子大小和代谢稳定性等多种因素共同影响。", defEn: "The percentage of a drug that reaches systemic circulation and the site of action. Influenced by solubility, pKa, molecular size, and metabolic stability." },
        { keys: ["官能团", "Functional Group"], en: "Functional Group", cn: "官能团", def: "分子中决定其化学性质的特定原子或原子团，如羟基 (-OH)、羧基 (-COOH)、氨基 (-NH2)、酯基 (-COOR) 等。官能团的种类和数量直接影响分子的溶解度、pKa 和反应活性。", defEn: "A specific atom or group of atoms that determines a molecule's chemical properties, e.g. hydroxyl (-OH), carboxyl (-COOH), amino (-NH2), ester (-COOR). The type and number of functional groups directly affect solubility, pKa, and reactivity." },
        { keys: ["氢键", "Hydrogen Bond", "H-Bond"], en: "Hydrogen Bond", cn: "氢键", def: "氢原子与电负性强的原子（O、N、F）之间的非共价相互作用。氢键是决定分子水溶性的最重要因素之一，也是维持蛋白质和 DNA 结构的关键力。", defEn: "A non-covalent interaction between a hydrogen atom and a strongly electronegative atom (O, N, F). Hydrogen bonds are among the most important factors governing water solubility and are critical for maintaining protein and DNA structure." },
        { keys: ["分子内氢键", "Intramolecular Hydrogen Bond", "Intramolecular H-Bond", "Intramolecular H Bond"], en: "Intramolecular Hydrogen Bond", cn: "分子内氢键", def: "同一分子内部不同基团之间形成的氢键。分子内氢键可稳定特定构象（如六元环螯合结构），屏蔽极性基团、降低表观极性（TPSA），从而提升膜渗透性和脂溶性。也可通过稳定共轭碱来显著调节 pKa 值。", defEn: "A hydrogen bond formed between different groups within the same molecule. Intramolecular H-bonds stabilize specific conformations (e.g. six-membered chelate rings), shield polar groups and reduce apparent polarity (TPSA), thereby enhancing membrane permeability and lipophilicity. They can also significantly modulate pKa by stabilizing the conjugate base." },
        { keys: ["诱导效应", "Inductive Effect"], en: "Inductive Effect", cn: "诱导效应", def: "电负性原子通过 σ 键对分子中其他原子产生电子吸引或排斥的现象。吸电子诱导效应 (-I) 可降低 pKa（增强酸性），推电子诱导效应 (+I) 可升高 pKa（减弱酸性）。", defEn: "The electron-attracting or -donating effect transmitted through σ bonds by electronegative atoms. Electron-withdrawing inductive effects (-I) lower pKa (increase acidity); electron-donating effects (+I) raise pKa (decrease acidity)." },
        { keys: ["共轭效应", "Resonance Effect", "Conjugation Effect", "Resonance", "Conjugation"], en: "Resonance / Conjugation Effect", cn: "共轭效应", def: "π 电子在共轭体系中离域分布的现象。共轭效应可稳定电离后的离子形式（如羧酸根负离子），从而增强酸性。对含有芳香环的分子尤为重要。", defEn: "The delocalization of π electrons across a conjugated system. Resonance can stabilize ionized forms (e.g. carboxylate anion), thereby increasing acidity. Particularly important for molecules containing aromatic rings." },
        { keys: ["空间位阻", "Steric Hindrance", "Steric Effect", "Steric"], en: "Steric Hindrance", cn: "空间位阻", def: "分子中体积较大的原子或基团阻碍化学反应的效应。空间位阻可影响质子的接近和离去，从而调节 pKa 值。", defEn: "The effect of bulky atoms or groups physically obstructing a chemical reaction. Steric hindrance can affect proton access and departure, thereby modulating pKa values." },
        { keys: ["杂化", "Hybridization", "杂化轨道", "杂化/芳香性"], en: "Hybridization", cn: "杂化轨道", def: "原子轨道线性组合形成新轨道的概念（sp、sp²、sp³）。杂化方式决定分子的几何构型和键角，影响电子分布。sp² 杂化的碳（如芳香环）比 sp³ 杂化的碳具有更强的吸电子能力。", defEn: "The concept of atomic orbitals combining linearly to form new orbitals (sp, sp², sp³). Hybridization determines molecular geometry and bond angles. sp²-hybridized carbon (e.g. in aromatic rings) is more electron-withdrawing than sp³-hybridized carbon." },
        { keys: ["肠溶片", "Enteric-Coated Tablet"], en: "Enteric-Coated Tablet", cn: "肠溶片", def: "一种特殊包衣的药物剂型，在胃酸中不溶解，到达小肠后才释放药物。用于保护胃黏膜或防止药物在酸性环境中降解。", defEn: "A drug dosage form with a special coating that resists stomach acid and releases the drug only upon reaching the small intestine. Used to protect the stomach lining or prevent drug degradation in acidic environments." },
        { keys: ["RDKit"], en: "RDKit", cn: "RDKit 化学信息学工具包", def: "开源的化学信息学软件库，用于分子结构的解析、化学描述符计算、分子指纹生成和结构绘制。本应用的核心化学计算均由 RDKit 驱动。", defEn: "An open-source cheminformatics toolkit for molecular structure parsing, descriptor calculation, fingerprint generation, and structure depiction. All core chemistry computations in this app are powered by RDKit." },
        { keys: ["Lipinski", "Lipinski's Rule of Five", "五规则", "Rule of Five", "Ro5"], en: "Lipinski's Rule of Five", cn: "Lipinski 五规则", def: "由 Pfizer 的 Christopher Lipinski 在 1997 年提出的口服药物筛选经验规则：分子量 ≤ 500、LogP ≤ 5、氢键供体 ≤ 5、氢键受体 ≤ 10。违反不超过 1 条的分子更可能具有良好的口服吸收。这是一条经验性初筛标准，不是绝对规则。", defEn: "Empirical oral drug-likeness rules proposed by Christopher Lipinski (Pfizer, 1997): MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10. Molecules violating ≤1 rule are more likely to have good oral absorption. This is an empirical screening guideline, not an absolute rule." },
        { keys: ["QED", "Quantitative Estimate of Drug-likeness", "药物相似性定量评估"], en: "QED (Quantitative Estimate of Drug-likeness)", cn: "QED 药物相似性定量评估", def: "由 Bickerton 等人 (2012) 提出的综合药物相似性指标，范围 0-1。QED 综合了分子量、LogP、氢键供体/受体数、TPSA、可旋转键数、芳香环数和结构警报，将分子映射到已知口服药物的化学空间。QED ≥ 0.67 = 有吸引力；0.49–0.67 = 中等；< 0.49 = 偏低。QED 的优势在于将多维性质整合为单一直观分数，便于快速筛选。", defEn: "A composite drug-likeness metric (0–1) proposed by Bickerton et al. (2012). QED integrates MW, LogP, HBD/HBA counts, TPSA, rotatable bonds, aromatic rings, and structural alerts, mapping molecules to the chemical space of known oral drugs. QED ≥ 0.67 = attractive; 0.49–0.67 = moderate; < 0.49 = low. Its strength lies in collapsing multidimensional properties into a single intuitive score for rapid screening." },
        { keys: ["SAscore", "Synthetic Accessibility Score", "合成可及性评分", "SA Score"], en: "SAscore (Synthetic Accessibility Score)", cn: "SAscore 合成可及性评分", def: "由 Ertl 和 Schuffenhauer (2009) 提出的合成难度评估指标，范围 1-10（1=极易合成，10=极难合成）。SAscore 基于两个核心部分：(1) 片段贡献 — 从 PubChem 数百万化合物中统计出的常见分子片段的“合成友好度”分数；(2) 复杂度惩罚 — 考虑分子大小、手性中心、螺环/桥环原子和大环结构。1-3 = 容易合成；3-6 = 中等难度；> 6 = 难以合成。该指标的计算与原始 PipelinePilot 实现的 R² = 0.97。", defEn: "A synthetic difficulty metric (1–10, 1=easiest, 10=hardest) proposed by Ertl & Schuffenhauer (2009). SAscore is based on two components: (1) Fragment contribution — 'synthetic friendliness' scores of common molecular fragments derived from millions of PubChem compounds; (2) Complexity penalty — accounting for molecular size, chiral centers, spiro/bridgehead atoms, and macrocycles. 1–3 = easy; 3–6 = moderate; > 6 = difficult. This implementation achieves R² = 0.97 with the original PipelinePilot implementation." },
        { keys: ["Fsp³", "Fsp3", "Fraction sp³", "Fraction sp³ Carbons", "Fraction sp3", "sp3碳比例", "三维复杂度"], en: "Fsp³ (Fraction of sp³ Carbons)", cn: "Fsp³ 三维复杂度指标", def: "由 Lovering 等人 (2009) 提出的分子三维复杂度指标，计算 sp³ 杂化碳原子占总碳原子数的比例，范围 0-1。研究发现 Fsp³ ≥ 0.45 的候选药物在临床试验中的成功率显著更高，原因包括：(1) 三维结构提高靶点选择性；(2) 减少平面芳香分子的 π-π 堆积导致的聚集和沉淀；(3) 增加溶解度和代谢稳定性。Fsp³ 越低，分子越'扁平'，更像传统筛选库中的分子；Fsp³ 越高，分子越接近天然产物的三维复杂性。", defEn: "A 3D complexity metric (0–1) proposed by Lovering et al. (2009), measuring the fraction of sp³-hybridized carbons among all carbon atoms. Clinical candidates with Fsp³ ≥ 0.45 show significantly higher success rates because: (1) 3D structures improve target selectivity; (2) Reduced π-π stacking (aggregation/precipitation) common in flat aromatic molecules; (3) Enhanced solubility and metabolic stability. Lower Fsp³ = flatter, more 'library-like'; higher Fsp³ = more natural-product-like 3D complexity." },
        { keys: ["ADME", "ADME/Tox", "ADMET", "药代动力学"], en: "ADME/Tox", cn: "ADME/Tox 药代动力学", def: "药物在体内的吸收 (Absorption)、分布 (Distribution)、代谢 (Metabolism)、排泄 (Excretion) 和毒性 (Toxicity) 五个关键过程。合称 ADME/Tox，是药物发现和开发中必须评估的核心性质。", defEn: "The five key processes governing a drug's fate in the body: Absorption, Distribution, Metabolism, Excretion, and Toxicity. ADME/Tox properties must be evaluated for any drug candidate during discovery and development." },
        { keys: ["CYP450", "CYP3A4", "CYP2D6", "CYP2C9", "细胞色素P450", "P450"], en: "Cytochrome P450 (CYP450)", cn: "细胞色素 P450 酶系", def: "肝脏中最重要的 I 相药物代谢酶超家族。CYP3A4 代谢约 50% 的临床药物；CYP2D6 存在显著的基因多态性；CYP2C9 参与多种 NSAID 药物代谢。药物间相互作用常与 CYP450 酶的诱导或抑制相关。", defEn: "The most important Phase I drug-metabolizing enzyme superfamily in the liver. CYP3A4 metabolizes ~50% of clinical drugs; CYP2D6 shows significant genetic polymorphism; CYP2C9 metabolizes many NSAIDs. Drug-drug interactions often involve CYP450 enzyme induction or inhibition." },
        { keys: ["血脑屏障", "Blood-Brain Barrier", "BBB"], en: "Blood-Brain Barrier (BBB)", cn: "血脑屏障", def: "由脑毛细血管内皮细胞紧密连接构成的选择性屏障，阻止大多数分子从血液进入大脑。低 TPSA (< 70 Å²)、适中 LogP、分子量较小的分子更容易通过血脑屏障。", defEn: "A selective barrier formed by tight junctions between brain capillary endothelial cells, blocking most molecules from entering the brain from blood. Molecules with low TPSA (< 70 Å²), moderate LogP, and smaller size are more likely to cross the BBB." },
        { keys: ["表观分布容积", "Volume of Distribution", "Vd", "分布容积"], en: "Volume of Distribution (Vd)", cn: "表观分布容积", def: "药物在体内的分布范围指标。Vd 越大，药物越倾向于分布到组织中；Vd 小则主要留在血浆中。亲脂性分子通常 Vd 较大。", defEn: "An indicator of a drug's distribution extent in the body. Large Vd means the drug prefers to distribute into tissues; small Vd means it mainly stays in plasma. Lipophilic molecules typically have larger Vd values." },
        { keys: ["血浆蛋白结合率", "Protein Binding", "Plasma Protein Binding", "PPB"], en: "Plasma Protein Binding", cn: "血浆蛋白结合率", def: "药物与血浆蛋白（主要是白蛋白和 α1-酸性糖蛋白）结合的比例。高结合率意味着药物缓慢释放、作用持久；只有游离药物能发挥药理作用。", defEn: "The fraction of drug bound to plasma proteins (mainly albumin and α1-acid glycoprotein). High binding means slow release and prolonged action; only free (unbound) drug exerts pharmacological effects." },
        { keys: ["结构警报", "Structural Alert", "毒性结构"], en: "Structural Alert", cn: "结构警报（毒性）", def: "历史上与毒性或不良反应相关的特定分子子结构（如硝基芳香族、芳香胺、环氧化物）。结构警报是药物设计初筛阶段的重要风险评估工具，但单一警报不等于分子一定有毒。", defEn: "Specific molecular substructures historically associated with toxicity or adverse effects (e.g. nitro-aromatics, aromatic amines, epoxides). Structural alerts are important risk assessment tools during early drug screening, but a single alert does not guarantee toxicity." },
        { keys: ["首过效应", "First-Pass Effect", "首过代谢"], en: "First-Pass Effect", cn: "首过效应", def: "口服药物经肠道吸收后首先通过肝脏，在进入体循环之前被肝脏代谢酶降解一部分。首过效应大的药物口服生物利用度低，可能需要更高剂量或改用其他给药途径。", defEn: "The metabolism of orally administered drugs by the liver before reaching systemic circulation. Drugs with high first-pass effects have low oral bioavailability and may require higher doses or alternative administration routes." },
        { keys: ["Michael 受体", "Michael Acceptor", "α,β-不饱和羰基"], en: "Michael Acceptor", cn: "Michael 受体", def: "含有 α,β-不饱和羰基结构的分子（C=C-C=O）。该结构可与生物亲核试剂（如谷胱甘肽、蛋白质半胱氨酸残基）发生 Michael 加成反应，可能引起非特异性毒性或过敏反应。", defEn: "A molecule containing an α,β-unsaturated carbonyl group (C=C-C=O). This structure can undergo Michael addition with biological nucleophiles (e.g. glutathione, protein cysteine residues), potentially causing non-specific toxicity or allergic reactions." }
    ];

    var GLOSSARY = {};
    var GLOSS_KEYS = [];
    for (var i = 0; i < GLOSS_ENTRIES.length; i++) {
        var e = GLOSS_ENTRIES[i];
        for (var j = 0; j < e.keys.length; j++) {
            GLOSSARY[e.keys[j]] = e;
            GLOSS_KEYS.push(e.keys[j]);
        }
    }
    // 按长度降序排列，确保长术语优先匹配
    GLOSS_KEYS.sort(function(a, b) { return b.length - a.length; });

    // ═══════════════════════════════════════════
    // 创建弹窗元素
    // ═══════════════════════════════════════════
    var popup = doc.getElementById('gloss-popup');
    if (!popup) {
        popup = doc.createElement('div');
        popup.id = 'gloss-popup';
        doc.body.appendChild(popup);
    }
    var overlay = doc.getElementById('gloss-overlay');
    if (!overlay) {
        overlay = doc.createElement('div');
        overlay.id = 'gloss-overlay';
        doc.body.appendChild(overlay);
    }

    function closePopup() {
        popup.classList.remove('show');
        overlay.classList.remove('show');
    }
    overlay.addEventListener('click', closePopup);
    doc.addEventListener('keydown', function(e) { if (e.key === 'Escape') closePopup(); });

    // ═══════════════════════════════════════════
    // 扫描页面，为术语添加点击事件
    // ═══════════════════════════════════════════
    function showPopup(term, x, y) {
        var entry = GLOSSARY[term];
        if (!entry) return;
        popup.innerHTML =
            '<div class="gloss-en">' + entry.en + '</div>' +
            '<div class="gloss-cn">' + entry.cn + '</div>' +
            '<div class="gloss-def">' + entry.def + '</div>' +
            '<div class="gloss-def-en">' + entry.defEn + '</div>';
        popup.classList.add('show');
        overlay.classList.add('show');

        // 定位：优先在点击位置右侧，空间不够则左侧
        var pw = popup.offsetWidth;
        var ph = popup.offsetHeight;
        var ww = win.innerWidth;
        var wh = win.innerHeight;
        var px = x + 18;
        var py = y - 20;
        if (px + pw > ww - 20) px = x - pw - 18;
        if (py + ph > wh - 20) py = wh - ph - 20;
        if (px < 10) px = 10;
        if (py < 10) py = 10;
        popup.style.left = px + 'px';
        popup.style.top = py + 'px';
    }

    // ═══════════════════════════════════════════
    // 递归收集文本节点（避免 TreeWalker/NodeFilter 跨上下文问题）
    // ═══════════════════════════════════════════
    function collectTextNodes(node, out) {
        if (node.nodeType === 3) { // TEXT_NODE
            var parent = node.parentNode;
            if (parent && !parent.closest('#gloss-popup') && !parent.closest('script') &&
                !parent.closest('style') && !parent.closest('input') && !parent.closest('textarea') &&
                !parent.closest('.gloss-term') && !parent.closest('code') && !parent.closest('pre') &&
                !parent.closest('[data-testid="stMetricValue"]'))
            {
                if (node.nodeValue && node.nodeValue.trim().length > 0) {
                    out.push(node);
                }
            }
        } else if (node.nodeType === 1 && node.childNodes && node.childNodes.length > 0) {
            // ELEMENT_NODE: 递归进入子节点（不进入 iframe）
            if (node.tagName !== 'IFRAME') {
                for (var c = 0; c < node.childNodes.length; c++) {
                    collectTextNodes(node.childNodes[c], out);
                }
            }
        }
    }

    function wrapTerms(root) {
        var textNodes = [];
        collectTextNodes(root, textNodes);
        console.log('[Glossary] Found ' + textNodes.length + ' text nodes to scan');

        var wrappedCount = 0;
        for (var i = 0; i < textNodes.length; i++) {
            var node = textNodes[i];
            var text = node.nodeValue;
            if (!text || !node.parentNode) continue;

            // 在文本中查找所有匹配的术语
            var matches = [];
            for (var k = 0; k < GLOSS_KEYS.length; k++) {
                var term = GLOSS_KEYS[k];
                var idx = -1;
                while ((idx = text.indexOf(term, idx + 1)) !== -1) {
                    matches.push({ idx: idx, term: term });
                }
            }
            if (matches.length === 0) continue;

            // 按位置升序，同位置优先长术语（避免"氢键"抢占"氢键供体"）
            matches.sort(function(a, b) { return a.idx !== b.idx ? a.idx - b.idx : b.term.length - a.term.length; });

            var frag = doc.createDocumentFragment();
            var lastIdx = 0;
            for (var m = 0; m < matches.length; m++) {
                var match = matches[m];
                if (match.idx < lastIdx) continue; // 重叠匹配跳过

                if (match.idx > lastIdx) {
                    frag.appendChild(doc.createTextNode(text.substring(lastIdx, match.idx)));
                }
                var entry = GLOSSARY[match.term];
                var span = doc.createElement('span');
                span.className = 'gloss-term';
                span.textContent = match.term;
                span.title = entry.en + ' — ' + entry.def.substring(0, 80) + '…';
                (function(t) {
                    span.addEventListener('click', function(e) {
                        e.stopPropagation();
                        showPopup(t, e.clientX, e.clientY);
                    });
                })(match.term);
                frag.appendChild(span);
                lastIdx = match.idx + match.term.length;
                wrappedCount++;
            }
            if (lastIdx < text.length) {
                frag.appendChild(doc.createTextNode(text.substring(lastIdx)));
            }
            node.parentNode.replaceChild(frag, node);
        }
        if (wrappedCount > 0) console.log('[Glossary] Wrapped ' + wrappedCount + ' terms');
    }

    // ═══════════════════════════════════════════
    // 初始扫描 + 动态监听
    // ═══════════════════════════════════════════
    var scanTimer = null;
    function scan() {
        if (scanTimer) return; // 防止短时间内重复扫描
        scanTimer = setTimeout(function() {
            scanTimer = null;
            try { wrapTerms(doc.body); } catch(e) { console.error('[Glossary] Error:', e); }
        }, 100);
    }
    scan();
    // Streamlit 动态渲染后重新扫描
    setTimeout(scan, 800);
    setTimeout(scan, 2000);
    setTimeout(scan, 4000);

    var obs = new MutationObserver(function() { scan(); });
    obs.observe(doc.body, { childList: true, subtree: true });
    console.log('[Glossary] Initialized — watching for terms');
})();
</script>"""


def inject_all_scripts():
    """Inject all client-side scripts in a single iframe for performance."""
    combined = (
        _DROPDOWN_OVERRIDE_JS + "\n" +
        _PARTICLE_STARRY_BG_JS + "\n" +
        _MOUSE_GLOW_TILT_JS + "\n" +
        _GLOSSARY_JS
    )
    _render_html(combined, height=1)
