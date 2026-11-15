// 보기(척도) 색을 만든다. 색은 OKLCH에서 밝기를 일정한 간격으로 나눠 만들고 sRGB 범위 안으로 맞춘다.
// 만든 색은 dataviz 검증 스크립트(validate_palette.js --ordinal)로 밝기 순서와 간격을 확인했다.

export type Theme = "light" | "dark";

function toSrgb(L: number, C: number, H: number): [number, number, number] {
  const h = (H * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
}

const gamma = (v: number) => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

/** OKLCH → #rrggbb. sRGB 밖이면 채도를 줄여 안으로 넣는다. */
export function oklchHex(L: number, C: number, H: number): string {
  let c = C;
  let rgb = toSrgb(L, c, H);
  while (c > 0 && rgb.some((v) => v < -0.0005 || v > 1.0005)) {
    c -= 0.005;
    rgb = toSrgb(L, Math.max(c, 0), H);
  }
  return (
    "#" +
    rgb
      .map((v) => Math.round(Math.min(1, Math.max(0, gamma(Math.min(1, Math.max(0, v))))) * 255))
      .map((v) => v.toString(16).padStart(2, "0"))
      .join("")
  );
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export interface Swatch {
  fill: string;
  /** 이 색 위에 올리는 글자색 */
  ink: string;
}

const INK_DARK = "#0b0b0b";
const INK_LIGHT = "#ffffff";
const HUE_LOW = 45; // 척도의 앞쪽(낮은 값): 따뜻한 색
const HUE_HIGH = 255; // 척도의 뒤쪽(높은 값): 파랑

// 가운데에서 끝으로 갈수록 진해진다(밝은 화면). 어두운 화면에서는 끝으로 갈수록 밝아진다.
const RANGE = {
  light: { near: 0.76, far: 0.4 },
  dark: { near: 0.42, far: 0.82 },
} as const;

function step(theme: Theme, hue: number, t: number): Swatch {
  const { near, far } = RANGE[theme];
  const L = lerp(near, far, t);
  const C = lerp(0.075, 0.15, t);
  return { fill: oklchHex(L, C, hue), ink: L > 0.64 ? INK_DARK : INK_LIGHT };
}

const NEUTRAL: Record<Theme, { mid: Swatch; off: Swatch }> = {
  light: {
    mid: { fill: "#e1e0d9", ink: INK_DARK }, // 척도의 가운데 ('보통')
    off: { fill: "#a9a79f", ink: INK_DARK }, // 척도 밖 ('잘 모르겠다', '기타')
  },
  dark: {
    mid: { fill: "#45443f", ink: INK_LIGHT },
    off: { fill: "#77756e", ink: INK_LIGHT },
  },
};

/**
 * 순서형 보기의 색.
 * 가운데가 있는 척도(유효한 보기가 홀수이고 5개 이상)는 가운데를 회색으로 두고 양쪽으로 갈리는 색을 쓴다.
 * 그렇지 않으면 한 가지 색이 한 방향으로 진해진다.
 * `values`가 null인 보기는 척도 밖이므로 회색으로 둔다.
 */
export function ordinalSwatches(values: (number | null)[], theme: Theme): Swatch[] {
  const valid = values.map((v, i) => ({ v, i })).filter((x) => x.v !== null) as { v: number; i: number }[];
  valid.sort((a, b) => a.v - b.v);
  const k = valid.length;
  const out: Swatch[] = values.map(() => NEUTRAL[theme].off);
  if (k % 2 === 1 && k >= 5) {
    const arm = (k - 1) / 2;
    valid.forEach((x, rank) => {
      const d = rank - arm; // 가운데에서 떨어진 칸 수
      if (d === 0) out[x.i] = NEUTRAL[theme].mid;
      else out[x.i] = step(theme, d < 0 ? HUE_LOW : HUE_HIGH, arm === 1 ? 1 : (Math.abs(d) - 1) / (arm - 1));
    });
  } else {
    valid.forEach((x, rank) => {
      out[x.i] = step(theme, HUE_HIGH, k === 1 ? 1 : rank / (k - 1));
    });
  }
  return out;
}

/** 화면에 쌓는 순서: 척도 순서대로, 척도 밖 보기는 맨 뒤. */
export function stackOrder(values: (number | null)[]): number[] {
  const idx = values.map((_, i) => i);
  const valid = idx.filter((i) => values[i] !== null).sort((a, b) => (values[a] as number) - (values[b] as number));
  return [...valid, ...idx.filter((i) => values[i] === null)];
}
