import json
import os
import urllib.request
import numpy as np


class Conflict:
    """A potential robot-robot conflict."""
    def __init__(self, i, j, dist, closing):
        self.i = i
        self.j = j
        self.dist = dist
        self.closing = closing


def detect_conflicts(robots, conflict_dist=1.5):
    """Return the first pair of robots that is closer than ``conflict_dist``
    and approaching each other (closing speed < 0)."""
    n = len(robots)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = robots[i], robots[j]
            dx, dy = rj.px - ri.px, rj.py - ri.py
            d = float(np.hypot(dx, dy))
            if d < 1e-6:
                continue
            closing = (dx * (rj.vx - ri.vx) + dy * (rj.vy - ri.vy)) / d
            if d < conflict_dist and closing < 0.0:
                return Conflict(i, j, d, closing)
    return None


class ArbitrationLayer:
    """Test-time high-level robot-robot arbitration.

    A conflict is detected when two robots are approaching within a distance
    threshold. The arbitrator decides which robot should yield (slow down) and
    returns a natural-language reason. The low-level RL policy still handles
    collision avoidance; the arbitration only modulates the forward velocity of
    the yielding robot, making the coordination visible and explainable.

    ``mode``:
      - 'llm'  : query a local Ollama model; fall back to rules on any failure.
      - 'rule' : deterministic rule (the robot farther from its goal yields).
    """
    def __init__(self, env, mode='llm', model='qwen2.5:7b',
                 ollama_url='http://localhost:11434/api/generate',
                 yield_factor=0.3, conflict_dist=1.5, clear_dist=2.2,
                 timeout=8):
        self.env = env
        self.mode = mode
        self.model = model
        self.ollama_url = ollama_url
        self.yield_factor = yield_factor
        self.conflict_dist = conflict_dist
        self.clear_dist = clear_dist
        self.timeout = timeout
        self.active_conflict = None      # (i, j, yielder, reason), latched
        self.reasons = []                # (step, yielder, reason)

    # ------------------------------------------------------------------ #
    def _apply_yield(self, actions):
        new_actions = [np.asarray(a, dtype=np.float32).copy() for a in actions]
        yielder = self.active_conflict[2]
        if yielder == 'both':
            for a in new_actions:
                a[0] *= self.yield_factor
        else:
            new_actions[yielder][0] *= self.yield_factor
        return new_actions

    def apply(self, actions):
        """Modulate actions based on arbitration. Returns (new_actions, reason)."""
        conflict = detect_conflicts(self.env.robots, self.conflict_dist)

        if conflict is None:
            if self.active_conflict is not None:
                # keep the decision latched until the conflicting pair is apart
                i, j, yielder, reason = self.active_conflict
                d = float(np.hypot(self.env.robots[i].px - self.env.robots[j].px,
                                   self.env.robots[i].py - self.env.robots[j].py))
                if d <= self.clear_dist:
                    return self._apply_yield(actions), reason
                self.active_conflict = None
            return actions, None

        if self.active_conflict is None:
            yielder, reason = self.decide(conflict)
            self.active_conflict = (conflict.i, conflict.j, yielder, reason)
            self.reasons.append((self.env.global_step, yielder, reason))
            print('[arbitrator] t=%.2fs %s' % (self.env.global_time, reason))
        return self._apply_yield(actions), self.active_conflict[3]

    # ------------------------------------------------------------------ #
    def decide(self, conflict):
        if self.mode == 'llm':
            try:
                return self._llm_decide(conflict)
            except Exception as e:
                print('[arbitrator] LLM call failed (%s), using rule fallback' % e)
                return self._rule_decide(conflict)
        return self._rule_decide(conflict)

    def _rule_decide(self, conflict):
        i, j = conflict.i, conflict.j
        di = self.env.robots[i].get_goal_distance()
        dj = self.env.robots[j].get_goal_distance()
        yielder, other = (i, j) if di > dj else (j, i)
        reason = ('rule: robot %d is farther from its goal (%.2fm vs %.2fm), '
                  'so it yields (distance %.2fm, closing %.2fm/s)' %
                  (yielder, di, dj, conflict.dist, -conflict.closing))
        return yielder, reason

    def _llm_decide(self, conflict):
        prompt = self._build_prompt(conflict)
        payload = json.dumps({'model': self.model, 'prompt': prompt,
                              'stream': False, 'format': 'json'}).encode()
        req = urllib.request.Request(self.ollama_url, data=payload,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode())
        data = self._parse_json(out.get('response', '{}'))
        y = data.get('yield')
        reason = str(data.get('reason', 'no reason given'))
        if y in (0, '0'):
            return 0, 'LLM: ' + reason
        if y in (1, '1'):
            return 1, 'LLM: ' + reason
        return 'both', 'LLM: ' + reason

    def _build_prompt(self, conflict):
        i, j = conflict.i, conflict.j
        lines = []
        lines.append('You coordinate two legged robots navigating in a social environment. '
                     'They are approaching each other and may collide. Decide which robot '
                     'should YIELD (slow down and let the other pass first) so that both '
                     'pass safely and politely.')
        for k in (i, j):
            r = self.env.robots[k]
            lines.append('Robot %d: position=(%.2f,%.2f) goal=(%.2f,%.2f) speed=%.2f m/s' %
                         (k, r.px, r.py, r.gx, r.gy, float(np.hypot(r.vx, r.vy))))
            nearby = []
            for h in self.env.humans:
                d = float(np.hypot(r.px - h.px, r.py - h.py))
                if d < self.env.laser_max_range:
                    nearby.append((round(float(h.emotion_value), 2), round(d, 2)))
            lines.append('  nearby pedestrians (emotion, distance): %s' % nearby)
        lines.append('distance between robots: %.2f m, closing speed: %.2f m/s' %
                     (conflict.dist, -conflict.closing))
        lines.append('Respond ONLY with JSON: {"yield": 0 or 1 or "both", '
                     '"reason": "one short sentence in English"}')
        return '\n'.join(lines)

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        if text.startswith('```'):
            text = text.split('```')[1] if '```' in text[3:] else text
            text = text.strip().lstrip('json').strip()
        try:
            return json.loads(text)
        except Exception:
            # try to find the first {...}
            start, end = text.find('{'), text.rfind('}')
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    pass
            return {}

    # ------------------------------------------------------------------ #
    def save_reasons(self, directory):
        if not self.reasons:
            return
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'arbitration_reasons.txt')
        with open(path, 'w') as f:
            for step, yielder, reason in self.reasons:
                f.write('step %d -> robot %s yields: %s\n' % (step, yielder, reason))
        print('[arbitrator] reasons saved to %s' % path)
