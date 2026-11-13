"""Hold scenes: the assistant stays silent while the user is busy or presenting.

Two scenes teach the duplex model that hearing speech is not a cue to talk:

* "wait"   - the user interrupts an answer ("can you wait a minute?"), the
  assistant acknowledges and goes silent while the user talks to other people
  or to themselves. For timed requests the assistant may check in when the
  time is up ("the minute is over, can I continue?"); the user can say "not
  yet, I'll tell you" and keep it waiting. When called back, the assistant
  resumes its interrupted answer where it stopped.
* "listen" - the user asks the assistant to listen until they finish (a
  presentation, a pitch, a speech rehearsal...). The assistant acknowledges,
  stays silent through the whole monologue including its pauses, and gives
  spoken feedback when the user says they are done.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

# (seconds, phrase) - the spoken duration always matches the real silence
TIMED_WAITS = [(10, "ten seconds"), (20, "twenty seconds"), (30, "half a minute"),
               (60, "a minute"), (90, "a minute and a half"), (120, "two minutes")]

HOLD_TIMED = [
    "Sorry, can you wait like {phrase}? I have to do something real quick.",
    "Hold on, give me {phrase}, someone needs me.",
    "Wait, can you pause for {phrase}? I need to take care of something.",
    "One second, actually, give me {phrase}, okay?",
]
HOLD_OPEN = [
    "Hang on, can you wait? I'll tell you when you can keep going.",
    "Hold that thought, I need to deal with something. I'll let you know.",
    "Wait, stop for a moment, I'll come back to you.",
    "Can you pause? Someone's at the door, I'll tell you when I'm back.",
]
ACK_WAIT = ["Sure, take your time.", "Of course, I'll wait.", "No problem, I'm here when you're ready.",
            "Okay, I'll hold on."]
CHECK_IN = ["Okay, the {phrase} is up. Should I continue?", "That was {phrase}. Can I go on?",
            "Alright, {phrase} has passed. Do you want me to continue?"]
NOT_YET = ["No, not yet. I'll let you know when you can start talking again.",
           "Not yet, please wait until I say so.", "Give me a bit longer, I'll tell you when."]
CALL_BACK = ["Okay, I'm back, you can continue.", "Alright, sorry about that. Go on.",
             "Okay, done. Where were we? Keep going.", "Okay, you can come back now."]

LISTEN_ASK = [
    "Can you just listen until I finish my {kind}? Don't interrupt me.",
    "I want to practice my {kind}. Please stay quiet until I'm done, then tell me what you think.",
    "Josie, just listen for a while, I'm going to rehearse my {kind}. I'll tell you when I'm finished.",
]
LISTEN_KINDS = ["presentation", "pitch", "talk", "speech", "project update", "lecture intro"]
ACK_LISTEN = ["Sure, I'm quiet now. Go ahead.", "Okay, I'm listening.", "Of course, take it away.",
              "Go for it, I won't interrupt."]
DONE = ["Okay Josie, I'm done! What do you think?", "That's it, I'm finished. How was it?",
        "Okay, done. Be honest, how did I do?"]

SIDE_TALK_PROMPT = (
    "Write {n} short separate things a person says out loud while busy with something else: "
    "to someone else in the room, on the phone, or muttering to themselves. They are NOT "
    "talking to an AI assistant. Loosely related situation: {topic}. Spoken style, 4-20 words "
    "each, one per line, no quotes, no numbering."
)
PRESENTATION_PROMPT = (
    "Write the spoken text of a {minutes}-minute {kind} someone rehearses out loud about: {topic}. "
    "Natural spoken style with a few filler words, a clear opening, 3-5 short parts, and a "
    "closing line. No headings, markdown or stage directions. Separate parts with blank lines."
)
FEEDBACK_NOTE = (
    "(The user just finished rehearsing the {kind} above while you listened silently. Give "
    "warm, honest spoken feedback: what worked, the one or two most important things to "
    "improve, and a concrete tip on delivery or structure. Keep it conversational, no lists or "
    "markdown, under 120 words.)"
)
RESUME_NOTE = (
    "(You were interrupted mid-answer and asked to wait; the user is now back. Continue your "
    "previous answer from exactly where you stopped, without repeating what you already "
    "said. You may start with a short bridge like 'Okay, so'.)"
)


@dataclass
class Clip:
    role: str
    text: str
    start: int  # sample
    speech: Any = None


def _lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*([-*\d.)]+\s*)", "", line).strip().strip('"')
        if 3 <= len(line.split()) <= 30:
            out.append(line)
    return out


class SceneBuilder:
    """Appends one scene to a generator's utterance timeline."""

    def __init__(self, gen: Any, rng: random.Random, history: list, utterances: list,
                 barge_ins: list, next_turn: int):
        self.gen, self.rng, self.history = gen, rng, history
        self.utterances, self.barge_ins = utterances, barge_ins
        self.cfg, self.rate = gen.config, gen.config.sample_rate
        self.turn = next_turn

    # ------------------------------------------------------------ helpers
    def ms(self, value: float) -> int:
        return round(value * self.rate / 1000)

    def secs(self, value: float) -> int:
        return round(value * self.rate)

    def say(self, role: str, text: str, start: int, history: bool = True):
        from synthetic.generator import Utterance

        speech = self.gen.speech.synthesize(text, role)
        self.gen._validate_speech(speech)
        utt = Utterance(role, self.turn, start, speech, speech)
        self.turn += 1
        self.utterances.append(utt)
        if history:
            self.history.append({"role": role, "content": text})
        return utt

    def ask_teacher(self, prompt: str) -> str:
        return self.gen.teacher.generate([{"role": "user", "content": prompt}],
                                         seed=self.rng.randrange(2**31))

    def side_talk(self, start: int, end: int, topic: str) -> int:
        """Fill [start, end) with user speech not addressed to the assistant."""
        n = max(1, int((end - start) / self.rate / 12))
        lines = _lines(self.ask_teacher(SIDE_TALK_PROMPT.format(n=n, topic=topic)))[:n]
        cursor = start + self.secs(self.rng.uniform(0.8, 2.5))
        last = start
        for line in lines:
            if cursor >= end - self.secs(1):
                break
            utt = self.say("user", line, cursor, history=False)
            last = utt.end_sample
            cursor = last + self.secs(self.rng.uniform(1.5, 6.0))
        return last

    def gap(self) -> int:
        return self.ms(self.cfg.response_gap_ms)

    # -------------------------------------------------------------- wait
    def wait(self, topic: str) -> dict:
        from synthetic.audio import stop_after_barge_in

        prev = self.utterances[-1]  # the assistant answer we interrupt
        onset = prev.speech.duration * self.rng.uniform(0.3, 0.7)
        onset = max(0, min(round(onset * self.rate), len(prev.speech.waveform) - 1)) / self.rate
        prev.speech = stop_after_barge_in(prev.speech, onset, self.cfg.assistant_stop_ms)
        self.history[-1]["content"] = prev.speech.spoken_text
        user_start = prev.start_sample + round(onset * self.rate)
        self.barge_ins.append({"user_turn": self.turn, "assistant_turn": prev.turn,
                               "onset_sample": user_start, "assistant_stop_sample": prev.end_sample,
                               "reaction_ms": (prev.end_sample - user_start) * 1000 / self.rate,
                               "cut_mid_word": prev.speech.cut_mid_word, "scene": "wait"})
        timed = self.rng.random() < 0.6
        allowed = [w for w in TIMED_WAITS if w[0] <= self.cfg.max_wait_seconds] or TIMED_WAITS[:1]
        seconds, phrase = self.rng.choice(allowed)
        request = self.rng.choice(HOLD_TIMED).format(phrase=phrase) if timed else self.rng.choice(HOLD_OPEN)
        req = self.say("user", request, user_start)
        ack = self.say("assistant", self.rng.choice(ACK_WAIT), req.end_sample + self.gap())
        wait_start = ack.end_sample
        if not timed:
            seconds = self.rng.uniform(8, min(45, self.cfg.max_wait_seconds))
        busy_end = self.side_talk(wait_start, wait_start + self.secs(seconds), topic)
        checked_in = timed and self.rng.random() < self.cfg.check_in_probability
        cursor = max(wait_start + self.secs(seconds), busy_end + self.secs(1))
        if checked_in:
            check = self.say("assistant", self.rng.choice(CHECK_IN).format(phrase=phrase), cursor)
            no = self.say("user", self.rng.choice(NOT_YET), check.end_sample + self.secs(0.6))
            extra = self.rng.uniform(6, max(8, self.cfg.max_wait_seconds / 2))
            busy_end = self.side_talk(no.end_sample, no.end_sample + self.secs(extra), topic)
            cursor = max(no.end_sample + self.secs(extra), busy_end + self.secs(1))
        back = self.say("user", self.rng.choice(CALL_BACK), cursor)
        messages = [dict(m) for m in self.history]
        messages[-1]["content"] += "\n\n" + RESUME_NOTE
        answer = self.gen.teacher.generate(messages, seed=self.rng.randrange(2**31))
        resume = self.say("assistant", answer, back.end_sample + self.gap())
        return {"type": "wait", "timed": timed, "wait_seconds": round(seconds, 2),
                "checked_in": checked_in, "start": user_start / self.rate,
                "silent_from": wait_start / self.rate, "resume": resume.start_sample / self.rate}

    # ------------------------------------------------------------ listen
    def listen(self, topic: str, start: int) -> dict:
        kind = self.rng.choice(LISTEN_KINDS)
        ask = self.say("user", self.rng.choice(LISTEN_ASK).format(kind=kind), start)
        ack = self.say("assistant", self.rng.choice(ACK_LISTEN), ask.end_sample + self.gap())
        minutes = self.rng.choice([1, 2, 3])
        text = self.ask_teacher(PRESENTATION_PROMPT.format(minutes=minutes, kind=kind, topic=topic))
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 4]
        cursor = ack.end_sample + self.secs(self.rng.uniform(0.8, 2.0))
        talk_start = cursor
        for part in parts:
            utt = self.say("user", part, cursor, history=False)
            cursor = utt.end_sample + self.secs(self.rng.uniform(0.8, 3.5))  # silent pauses
        done = self.say("user", self.rng.choice(DONE), cursor)
        self.history[-1]["content"] = "\n\n".join(parts) + "\n\n" + self.history[-1]["content"]
        messages = [dict(m) for m in self.history]
        messages[-1]["content"] += "\n\n" + FEEDBACK_NOTE.format(kind=kind)
        answer = self.gen.teacher.generate(messages, seed=self.rng.randrange(2**31))
        reply = self.say("assistant", answer, done.end_sample + self.gap())
        return {"type": "listen", "kind": kind, "parts": len(parts), "start": start / self.rate,
                "monologue_seconds": (done.start_sample - talk_start) / self.rate,
                "feedback": reply.start_sample / self.rate}

    # -------------------------------------------------------- backchannel
    def backchannel(self, user_utt: Any, not_before: int) -> list[dict]:
        """Short assistant acknowledgements in the pauses of a long user turn.

        They never take the floor: placed 150-450 ms after a word boundary with
        a pause, at least 1.5 s apart, and finished before the user stops.
        They are not added to the dialogue history.
        """
        from synthetic.audio import SpeechAlignmentError

        words = user_utt.speech.words
        pauses = [w.end for w, nxt in zip(words, words[1:])
                  if nxt.start - w.end >= 0.12 or w.word.endswith((",", ".", "?", "!"))]
        self.rng.shuffle(pauses)
        placed, out = [], []
        budget = self.rng.choice([1, 1, 2, 3])
        for t in sorted(pauses[: budget * 3]):
            if len(out) >= budget:
                break
            start = user_utt.start_sample + self.secs(t + self.rng.uniform(0.15, 0.45))
            if start < not_before or any(abs(start - p) < self.secs(1.5) for p in placed):
                continue
            try:
                utt = self.say("assistant", self.rng.choice(BACKCHANNELS), start, history=False)
            except SpeechAlignmentError:
                continue
            if utt.end_sample >= user_utt.end_sample:  # would hold the floor: drop it
                self.utterances.pop()
                continue
            placed.append(start)
            out.append({"start": start / self.rate, "text": utt.speech.source_text})
        return out


BACKCHANNELS = ["Mhm.", "Yeah.", "Right.", "Okay.", "Uh-huh.", "I see.", "Got it.", "Sure.", "Yeah, yeah."]
