"""Monotonic stage budgets for one integrated respondent turn."""
import time
from threading import Lock
from respondent_core import StudioError

class TurnBudget:
    def __init__(self,retrieval=240,answer=300):
        self.retrieval=float(retrieval);self.answer=float(answer)
        self.started=time.monotonic();self.phase="retrieval"
        self.deadline=self.started+self.retrieval
        self.events=[];self.lock=Lock()
    def start_answer(self,relief=0):
        self.check()
        self.phase="answer"
        self.deadline=time.monotonic()+self.answer+relief
    def remaining(self):
        return self.deadline-time.monotonic()
    def check(self):
        if self.remaining()<=0:
            exc=StudioError("受访者"+("检索" if self.phase=="retrieval" else "回答")+"阶段超时；本轮未提交，可手动继续。",
                "respondent_stage_timeout",503,detail_code=self.phase+"_deadline",retryable=True)
            exc.response_structure=self.snapshot()
            raise exc
    def limit(self,requested):
        self.check()
        return max(0.001,min(float(requested),self.remaining()))
    def record(self,phase,elapsed,ok,attempts=None):
        with self.lock:
            self.events.append({"stage":phase,"elapsed_ms":round(elapsed*1000),"ok":ok,"attempts":attempts})
    def snapshot(self):
        with self.lock:
            return {"retrieval_budget_seconds":self.retrieval,"answer_budget_seconds":self.answer,
                    "phase":self.phase,"elapsed_ms":round((time.monotonic()-self.started)*1000),
                    "requests":list(self.events)}
