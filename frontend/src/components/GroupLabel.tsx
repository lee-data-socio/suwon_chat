import type { GroupRow } from "../types";
import { count } from "../lib/format";
import { Hint } from "./Tooltip";

/** 집단 이름 옆에 붙는 표시: 표본이 적은 집단, 학습에서 뺀 집단 */
export function GroupBadges({ g }: { g: GroupRow }) {
  return (
    <>
      {g.small_sample && (
        <Hint
          className="badge badge-warn"
          text="응답자가 100명 이하다. 실측 분포 자체가 표본에 따라 흔들리므로 이 집단의 오차는 크게 읽지 않는다."
        >
          표본 적음
        </Hint>
      )}
      {g.held_out && (
        <Hint className="badge badge-held" text="이 어댑터는 이 집단의 응답을 한 번도 보지 않고 학습했다.">
          학습에서 뺌
        </Hint>
      )}
    </>
  );
}

export function GroupLabel({ g }: { g: GroupRow }) {
  return (
    <div className="group-label">
      <div className="group-name" title={g.group}>
        {g.group}
      </div>
      <div className="group-sub">
        <span className="group-n">{count(g.n)}</span>
        <GroupBadges g={g} />
      </div>
    </div>
  );
}
