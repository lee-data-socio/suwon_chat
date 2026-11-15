import type { AdapterInfo, LiveInfo } from "../types";

// 어댑터 이름은 앱 서버(/api/meta)가 정한다. 화면은 이름을 직접 적지 않고 여기서 받아 쓴다.
let adapters: AdapterInfo[] = [];

/** 집단·문항을 빼고 학습한 어댑터를 견줄 기준 어댑터. 앱 서버가 알려 준다. */
export let REFERENCE_ID = "exp1";

export function setAdapters(list: AdapterInfo[], referenceId?: string) {
  adapters = list;
  if (referenceId) REFERENCE_ID = referenceId;
}

export function adapterLabel(id: string): string {
  return adapters.find((a) => a.id === id)?.label ?? id;
}

/** 좁은 자리에 쓰는 짧은 이름: '전체 학습 어댑터' → '전체 학습' */
export function adapterShort(id: string): string {
  return adapterLabel(id).replace(/\s*어댑터$/, "");
}

// 실시간 모드에서 돌리고 있는 모델. 앱 서버(/api/meta.live)가 알려 준다.
let liveInfo: LiveInfo | null = null;
let liveMode = false;

export function setLive(info: LiveInfo | null | undefined, live: boolean) {
  liveInfo = info ?? null;
  liveMode = live;
}

/** 앱 서버가 예측을 그때그때 모델을 돌려 만드는가 */
export const isLiveMode = () => liveMode;

/** '사전 계산'·'실시간' 표시에 마우스를 올렸을 때 나오는 설명 */
export function sourceTitle(live: boolean): string {
  if (!live) return "미리 계산해 저장해 둔 예측이다. 모델을 지금 돌린 값이 아니다.";
  if (!liveInfo) return "모델을 지금 돌려 얻은 예측이다.";
  const name = liveInfo.base_model.split("/").filter(Boolean).pop() ?? liveInfo.base_model;
  const parts = [name, liveInfo.device, liveInfo.four_bit ? "4비트" : liveInfo.dtype, `어댑터 ${liveInfo.adapters.length}개`];
  return `모델을 지금 돌려 얻은 예측이다. ${parts.join(" · ")}`;
}
