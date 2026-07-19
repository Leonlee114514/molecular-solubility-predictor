import { useEffect, useRef } from "react";

/**
 * Deep-space canvas backdrop ported from assets/scripts.py
 * (_PARTICLE_STARRY_BG_JS): ~200 stars in 3 parallax layers with twinkle,
 * mouse attraction and local constellation lines, over 4 radial nebula
 * gradients. Mouse trail / click bursts from the old app are intentionally
 * omitted. Honors prefers-reduced-motion with a single static frame.
 */

interface Star {
  x: number;
  y: number;
  r: number;
  baseR: number;
  vx: number;
  vy: number;
  color: string;
  alpha: number;
  twinkleSpeed: number;
  twinklePhase: number;
  layer: number;
}

const LAYERS = [
  {
    count: 50,
    minR: 1.8,
    maxR: 3.5,
    speed: 0.08,
    colors: ["rgba(196,181,253,", "rgba(103,232,249,", "rgba(233,213,255,", "rgba(255,255,255,"],
  },
  {
    count: 70,
    minR: 1.0,
    maxR: 2.2,
    speed: 0.04,
    colors: ["rgba(167,139,250,", "rgba(34,211,238,", "rgba(139,92,246,", "rgba(253,224,71,"],
  },
  {
    count: 80,
    minR: 0.5,
    maxR: 1.2,
    speed: 0.02,
    colors: ["rgba(196,181,253,", "rgba(103,232,249,", "rgba(255,255,255,"],
  },
];

export default function StarfieldBackdrop() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let W = window.innerWidth;
    let H = window.innerHeight;
    let frame = 0;
    let rafId = 0;
    let mouseX = W / 2;
    let mouseY = H / 2;

    const resize = () => {
      W = canvas.width = window.innerWidth;
      H = canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const stars: Star[] = [];
    for (const [li, layer] of LAYERS.entries()) {
      for (let i = 0; i < layer.count; i++) {
        const r = layer.minR + Math.random() * (layer.maxR - layer.minR);
        stars.push({
          x: Math.random() * W,
          y: Math.random() * H,
          r,
          baseR: r,
          vx: (Math.random() - 0.5) * layer.speed,
          vy: (Math.random() - 0.5) * layer.speed,
          color: layer.colors[Math.floor(Math.random() * layer.colors.length)],
          alpha: 0.5 + Math.random() * 0.5,
          twinkleSpeed: 0.005 + Math.random() * 0.015,
          twinklePhase: Math.random() * Math.PI * 2,
          layer: li,
        });
      }
    }

    const drawBackground = () => {
      const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
      bgGrad.addColorStop(0, "#0f0f1c");
      bgGrad.addColorStop(0.3, "#131328");
      bgGrad.addColorStop(0.5, "#181830");
      bgGrad.addColorStop(0.7, "#131328");
      bgGrad.addColorStop(1, "#0f0f1c");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      const nebulas = [
        { x: W * 0.18, y: H * 0.25, rx: W * 0.35, ry: H * 0.28, color: "rgba(124,58,237," },
        { x: W * 0.82, y: H * 0.12, rx: W * 0.28, ry: H * 0.22, color: "rgba(6,182,212," },
        { x: W * 0.5, y: H * 0.8, rx: W * 0.25, ry: H * 0.2, color: "rgba(251,191,36," },
        { x: W * 0.5, y: H * 0.4, rx: W * 0.4, ry: H * 0.32, color: "rgba(67,56,202," },
      ];
      for (const n of nebulas) {
        const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, Math.max(n.rx, n.ry));
        g.addColorStop(0, `${n.color}0.25)`);
        g.addColorStop(0.5, `${n.color}0.08)`);
        g.addColorStop(1, `${n.color}0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.ellipse(n.x, n.y, n.rx, n.ry, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const drawStars = (animate: boolean) => {
      // Constellation lines near the cursor.
      let lineCount = 0;
      const maxLines = 30;
      for (let i = 0; i < stars.length && lineCount < maxLines; i++) {
        const dmx = stars[i].x - mouseX;
        const dmy = stars[i].y - mouseY;
        const dMouse = Math.hypot(dmx, dmy);
        if (dMouse > 200) continue;
        for (let j = i + 1; j < stars.length && lineCount < maxLines; j++) {
          if (stars[j].layer > 1) continue;
          const dist = Math.hypot(stars[i].x - stars[j].x, stars[i].y - stars[j].y);
          if (dist < 120) {
            const lineAlpha = 0.1 * (1 - dist / 120) * (1 - dMouse / 200);
            ctx.beginPath();
            ctx.moveTo(stars[i].x, stars[i].y);
            ctx.lineTo(stars[j].x, stars[j].y);
            ctx.strokeStyle = `rgba(167,139,250,${lineAlpha})`;
            ctx.lineWidth = 0.6;
            ctx.stroke();
            lineCount++;
          }
        }
      }

      for (const s of stars) {
        if (animate) {
          s.x += s.vx;
          s.y += s.vy;
          s.vx *= 0.992;
          s.vy *= 0.992;
          const mdx = mouseX - s.x;
          const mdy = mouseY - s.y;
          const mDist = Math.hypot(mdx, mdy);
          if (mDist < 250 && mDist > 5) {
            const attract = ((250 - mDist) / 250) * 0.35 * (1 - s.layer * 0.3);
            s.x += (mdx / mDist) * attract;
            s.y += (mdy / mDist) * attract;
          }
          if (s.x < -20) s.x = W + 20;
          if (s.x > W + 20) s.x = -20;
          if (s.y < -20) s.y = H + 20;
          if (s.y > H + 20) s.y = -20;
        }

        const mDist = Math.hypot(mouseX - s.x, mouseY - s.y);
        const twinkle = Math.sin(frame * s.twinkleSpeed + s.twinklePhase);
        let curAlpha = s.alpha * (0.5 + 0.5 * twinkle);
        s.r = s.baseR * (0.8 + 0.2 * twinkle);
        if (mDist < 200) {
          curAlpha = Math.min(1, curAlpha + ((200 - mDist) / 200) * 0.4);
          s.r *= 1 + ((200 - mDist) / 200) * 0.5;
        }
        const glowR = s.r * (s.layer === 0 ? 8 : 5);
        const glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, glowR);
        glow.addColorStop(0, `${s.color}${curAlpha * 0.5})`);
        glow.addColorStop(1, `${s.color}0)`);
        ctx.beginPath();
        ctx.arc(s.x, s.y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `${s.color}${curAlpha})`;
        ctx.fill();
      }

      if (animate) {
        const mGlow = ctx.createRadialGradient(mouseX, mouseY, 0, mouseX, mouseY, 60);
        mGlow.addColorStop(0, "rgba(196,181,253,0.15)");
        mGlow.addColorStop(0.4, "rgba(124,58,237,0.06)");
        mGlow.addColorStop(1, "rgba(124,58,237,0)");
        ctx.beginPath();
        ctx.arc(mouseX, mouseY, 60, 0, Math.PI * 2);
        ctx.fillStyle = mGlow;
        ctx.fill();
      }
    };

    if (reduced) {
      // Static frame only — no motion.
      drawBackground();
      drawStars(false);
      window.removeEventListener("resize", resize);
      const onResizeStatic = () => {
        resize();
        drawBackground();
        drawStars(false);
      };
      window.addEventListener("resize", onResizeStatic);
      return () => window.removeEventListener("resize", onResizeStatic);
    }

    const onMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };
    document.addEventListener("mousemove", onMouseMove, { passive: true });

    const animate = () => {
      frame++;
      drawBackground();
      drawStars(true);
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);

    const onVisibility = () => {
      if (document.hidden) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      } else if (!rafId) {
        rafId = requestAnimationFrame(animate);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-0 h-full w-full"
      aria-hidden="true"
    />
  );
}
