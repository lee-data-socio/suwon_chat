import type { AdapterInfo } from "../types";

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
