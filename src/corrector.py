"""
corrector.py — Probabilistic Character-Level N-gram Correction Layer
=====================================================================
Fills [?] gaps produced by the DSP decoder and corrects common Morse
confusions (E↔I, S↔H, T↔M, etc.) using an interpolated trigram language
model trained on a bundled English + amateur-radio corpus.

Model
-----
  Interpolated trigram → bigram → unigram, Laplace-smoothed.
  Weights: λ₁=0.10 (unigram), λ₂=0.30 (bigram), λ₃=0.60 (trigram).

Two correction passes
---------------------
  1. [?] fill  — bidirectional scoring: forward N-gram on left context +
                 backward N-gram (trained on reversed corpus) on right context.
  2. Confusion fix — for each character in a Morse confusion set, swap to the
                 most probable partner only when it is ≥ CONF_THRESHOLD × more
                 likely given the preceding bigram/trigram context.

Usage
-----
    from src.corrector import MorseCorrector
    c = MorseCorrector()
    clean = c.correct("HE[?]LO W[?]RLD")   # → "HELLO WORLD"

    # or module-level singleton:
    from src.corrector import correct
    clean = correct("CQ [?]E W1ABC")
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set


# ── Morse symbol confusion sets ───────────────────────────────────────────────
# Characters whose Morse patterns differ by exactly one element (dot/dash) and
# are therefore the most likely substitution errors in a noisy decode.
#
#   E  ( . )     ↔  I  ( .. )   — 1 vs 2 dots
#   I  ( .. )    ↔  S  ( ... )  — 2 vs 3 dots
#   S  ( ... )   ↔  H  ( .... ) — 3 vs 4 dots
#   T  ( - )     ↔  N  ( -. )   — 1 dash vs dash-dot
#   T  ( - )     ↔  M  ( -- )   — 1 vs 2 dashes
#   M  ( -- )    ↔  K  ( -.- )  — dashes vs dash-dot-dash
#   N  ( -. )    ↔  A  ( .- )   — dash-dot vs dot-dash
#   A  ( .- )    ↔  W  ( .-- )  — dot-dash vs dot-dash-dash
#   K  ( -.- )   ↔  C  ( -.-. ) — dash-dot-dash vs dash-dot-dash-dot
#   4  ( ....- ) ↔  5  ( ..... ) — four dots-dash vs five dots
#   5  ( ..... ) ↔  H  ( .... ) — five dots vs four dots
#   3  ( ...-- ) ↔  4  ( ....- ) — three-dot-two-dash vs four-dot-one-dash

_CONFUSION_SETS: List[frozenset] = [
    frozenset({"E", "I"}),
    frozenset({"I", "S"}),
    frozenset({"S", "H"}),
    frozenset({"T", "N"}),
    frozenset({"T", "M"}),
    frozenset({"M", "K"}),
    frozenset({"N", "A"}),
    frozenset({"A", "W"}),
    frozenset({"K", "C"}),
    frozenset({"4", "5"}),
    frozenset({"5", "H"}),
    frozenset({"3", "4"}),
]

_CONFUSION_MAP: Dict[str, Set[str]] = defaultdict(set)
for _cs in _CONFUSION_SETS:
    for _c in _cs:
        _CONFUSION_MAP[_c] |= _cs - {_c}


# ── English + Amateur Radio training corpus ───────────────────────────────────
# ~5 000 characters, all uppercase (Morse decodes to uppercase).
# Mix of high-coverage English prose, common English words, and ham-radio QSO
# language so the model knows both standard and on-air vocabulary.

_CORPUS = (
    # ── Pangrams & test sentences (full-alphabet coverage) ────────────────────
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG "
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS "
    "HOW VEXINGLY QUICK DAFT ZEBRAS JUMP "
    "THE FIVE BOXING WIZARDS JUMP QUICKLY "
    "SPHINX OF BLACK QUARTZ JUDGE MY VOW "
    "JACKDAWS LOVE MY BIG SPHINX OF QUARTZ "
    "AMAZINGLY FEW DISCOTHEQUES PROVIDE JUKEBOXES "
    "BLOWZY NIGHT FRUMPS VEXED JACK Q "
    "SIXTY ZIPPERS WERE QUICKLY PICKED FROM THE WOVEN JUTE BAG "
    # ── General English prose ─────────────────────────────────────────────────
    "THE SIGNAL IS STRONG AND CLEAR TODAY "
    "GOOD MORNING THIS IS A TEST TRANSMISSION "
    "THE WEATHER IS FINE AND THE VISIBILITY IS GOOD "
    "PLEASE CONFIRM RECEIPT THE MESSAGE HAS BEEN RECEIVED "
    "STANDING BY FOR YOUR REPLY ON THIS FREQUENCY "
    "WE WILL MEET AT THE AGREED TIME AND PLACE "
    "THE STATION IS OPERATING ON SEVEN MEGAHERTZ "
    "CALLING ALL STATIONS THIS IS AN IMPORTANT NOTICE "
    "ROGER THAT COPY LOUD AND CLEAR OVER AND OUT "
    "MESSAGE RECEIVED AND UNDERSTOOD ACKNOWLEDGED "
    "THE TRANSMISSION IS CLEAR AND READABLE "
    "FREQUENCY IS ON THE DOT CARRIER STABLE "
    "THE ANTENNA IS WORKING WELL AT ONE HUNDRED WATTS "
    "LONG DISTANCE COMMUNICATION IS POSSIBLE ON HIGH FREQUENCY "
    # ── Common English function words (high bigram density) ───────────────────
    "THE AND FOR ARE BUT NOT YOU ALL CAN HER WAS ONE OUR OUT "
    "DAY GET HAS HIM HIS HOW MAN NEW NOW OLD SEE TWO WAY WHO "
    "BOY DID ITS LET PUT SAY SHE TOO USE MAY TRY ASK SET "
    "ABOUT AFTER AGAIN COME COULD DOWN FIND FIRST GIVE GOOD "
    "GREAT HAND HERE HIGH HOLD JUST KEEP KIND KNOW LAST LATE "
    "LIFE LIKE LONG LOOK MADE MAKE MANY MOST MOVE MUCH NEED "
    "NEXT ONLY OPEN OVER PART PLAY POINT PULL PUSH REAL SAME "
    "SEEM SHOW SIDE SOME SOON TAKE TELL THAN THAT THEM THEN "
    "THERE THEY THIS TIME TURN VERY WANT WATER WELL WERE WHAT "
    "WHEN WHERE WHICH WHILE WITH WORD WORK WOULD YEAR YOUR "
    "ALSO BACK BEEN BOTH CALL CASE COME DOES EACH EVEN FACE "
    "FACT FEEL FREE FROM FULL GONE HALF HARD HEAD HELD HELP "
    "HOME HOPE HOUR IDEA INTO KEPT KNEW LEFT LESS LINE LIVE "
    "LOST LOVE MEAN MEET MIND MORE MOVE MUST NAME NEAR NEXT "
    "ONCE ONES ONLY OPEN PAST PATH PLAN POOR READ REST RICH "
    "ROAD ROOM ROSE RULE SAID SEEN SELF SENT SLOW SOON SORT "
    "STAY STEP STOP SUCH SURE TEST THEY THUS TOLD TOOK TOWN "
    "TREE TRUE TURN TYPE UNIT USED USES VIEW WALK WENT WEST "
    "WIDE WIFE WILD WILL WIND WISE WISH WISH WITH WOKE WOOD "
    "WORE WORN YEAR ZERO "
    # ── Common English letter clusters (explicit bigram/trigram reinforcement) -
    "THE THEN THERE THEIR THESE THEY THIN THINK THIRD THOSE "
    "THAT THAN THANK THING THROUGH THREE OTHER ANOTHER "
    "AND HAND LAND BAND SAND STAND GRAND BRAND BLAND "
    "ING RING SING KING BRING SPRING STING STRING THING "
    "ION STATION NATION RATION OPTION SECTION QUESTION "
    "TION ACTION FICTION MENTION FUNCTION ATTENTION "
    "ERA GENERAL SEVERAL FEDERAL MINERAL LIBERAL CENTRAL "
    "ERE HERE THERE WHERE WERE SEVERE EXTREME EXTREME "
    "EST BEST REST TEST WEST NEST FEST GUEST CHEST BREAST "
    "ALL CALL FALL HALL TALL WALL BALL CALL SMALL STALL "
    "OUR FOUR YOUR HOUR FLOUR SCOUR COLOUR HONOUR "
    "OVE LOVE MOVE ABOVE PROVE GROVE SHOVE STOVE WOVE "
    "AVE HAVE GAVE CAVE SAVE WAVE BRAVE GRAVE PAVE SHAVE "
    "ATE LATE RATE GATE HATE FATE MATE PLATE STATE SKATE "
    "AID SAID PAID RAID LAID MAID BRAID AFRAID "
    "AIR FAIR HAIR PAIR CHAIR STAIR REPAIR AFFAIR "
    "EAD READ HEAD LEAD DEAD BREAD DREAD SPREAD THREAD "
    "EAR HEAR NEAR YEAR DEAR FEAR GEAR CLEAR SPEAR SWEAR "
    "EAT HEAT MEAT BEAT FEAT SEAT TREAT WHEAT CHEAT "
    "EEN BEEN SEEN KEEN QUEEN SCREEN BETWEEN SIXTEEN "
    "ELL BELL CELL FELL SELL TELL WELL SPELL SMELL SHELL "
    "END BEND LEND MEND SEND TEND BLEND SPEND TREND "
    "ENT BENT DENT LENT RENT SENT TENT VENT WENT SPENT "
    "ERN FERN KERN STERN INTERN MODERN CONCERN PATTERN "
    "ERT BERT CERT PERT ALERT EXPERT CONVERT DESERT "
    "EST BEST NEST REST TEST VEST WEST CREST QUEST CHEST "
    # ── Numbers common in radio communications ────────────────────────────────
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE ZERO TEN "
    "ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN "
    "EIGHTEEN NINETEEN TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY "
    "EIGHTY NINETY HUNDRED THOUSAND MILLION "
    "10 20 30 40 50 60 70 80 90 100 200 300 500 1000 "
    "14025 14100 21050 28500 144200 432100 7050 3525 "
    "73 88 55 33 RST 599 579 449 339 229 "
    # ── Amateur radio / ham-radio QSO language ────────────────────────────────
    "CQ CQ CQ DE W1ABC W1ABC W1ABC K "
    "QRZ THIS IS W1ABC CALLING CQ DX K "
    "QTH NEW ENGLAND RST 599 NAME JOHN AGE 45 "
    "RIG ICOM IC7300 ANT DIPOLE PWR 100 WATTS "
    "QSL VIA BUREAU 73 DE W1ABC SK "
    "CQ DX CQ DX DE VK2XYZ VK2XYZ K "
    "QRM LOCAL STATION QRN STATIC NOISE QSB SIGNAL FADING "
    "QRP LOW POWER OPERATION QRO HIGH POWER AMPLIFIER "
    "QSO THIS CONTACT IS CONFIRMED LOGGED DIRECT BUREAU "
    "QSL CARD RECEIVED THANK YOU FOR THE CONTACT "
    "QSY MOVE TO NEW FREQUENCY QRL FREQUENCY IN USE "
    "QRX PLEASE WAIT QRV READY TO OPERATE "
    "CONTEST NUMBER 001 ZONE 8 STATE MASSACHUSETTS "
    "5NN 599 QSO COMPLETE 73 OLD MAN GOOD DX "
    "DE K7XYZ UR RST 579 QTH SEATTLE WASHINGTON "
    "TNX FB SIG WX CLOUDY 15 DEGREES CELSIUS "
    "GOING QRT NOW 73 73 DE W1ABC SK SK "
    "CW OPERATOR KEYER SPEED 25 WPM IAMBIC PADDLE "
    "MORSE CODE TRANSMISSION CLEAR AND READABLE "
    "ALPHA 14025 KHZ 20 METER BAND CALLING CQ "
    "VHF UHF REPEATER FREQUENCY OFFSET CTCSS TONE "
    "EMERGENCY COMMUNICATIONS NET CHECKIN 40 METERS "
    "NET CONTROL THIS SESSION ALL STATIONS STANDBY "
    "BREAK BREAK PRIORITY TRAFFIC FOR NET CONTROL "
    "RELAY MESSAGE TO ALL STATIONS PLEASE "
    "SIGNAL REPORT IS 5 AND 9 FULL QUIETING "
    "CALLSIGN NOVEMBER FOXTROT SEVEN INDIA BRAVO TANGO "
    "PHONETIC ALPHABET USED IN RADIO COMMUNICATIONS "
    "INTERNATIONAL MORSE CODE DISTRESS SIGNAL SOS "
    "MAYDAY MAYDAY MAYDAY POSITION BEARING LATITUDE "
    "VESSEL NAME SOULS ON BOARD NATURE OF DISTRESS "
    "COAST GUARD STATION CONTACT US ON CHANNEL 16 "
    "AMATEUR RADIO EMERGENCY SERVICE ARES RACES "
    "FREQUENCY COORDINATION REPEATER TRUSTEE CALL "
    "DE W6ABC QTH LOS ANGELES CALIFORNIA GRID DM04 "
    "DE G3XYZ QTH LONDON ENGLAND GRID IO91 "
    "DE JA1ABC QTH TOKYO JAPAN RST 599 559 339 "
    "DIPOLE YAGI BEAM VERTICAL LOOP ANTENNA SYSTEM "
    "KILOWATT MILLIWATT DECIBEL GAIN LOSS SIDEBAND "
    "UPPER SIDEBAND LOWER SIDEBAND AMPLITUDE FREQUENCY "
    "MODULATION DEMODULATION CARRIER WAVE BANDWIDTH "
    "TRANSCEIVER RECEIVER TRANSMITTER MICROPHONE "
    "HEADSET SPEAKER ANTENNA TUNER MATCHING NETWORK "
    "PROPAGATION IONOSPHERE SKIP DISTANCE GREYLINE "
    "SUNSPOT CYCLE SOLAR FLUX GEOMAGNETIC STORM "
    "DX PEDITION RARE ENTITY CALLSIGN PILE UP "
    "CONTEST EXCHANGE SERIAL NUMBER ZONE STATE "
    "CERTIFICATE AWARD DXCC MARATHON WORKED ALL STATES "
    "CLUB STATION FIELD DAY EMERGENCY POWER GENERATOR "
    "LICENSE TECHNICIAN GENERAL AMATEUR EXTRA CLASS "
    # ── Ham prosigns and common abbreviations ─────────────────────────────────
    "CQ DE K KN AR SK BT AS VE "
    "HI HI OM YL XYL FB UR PSE TNX "
    "GA GE GM GN GM GA CUL "
    "WX RIG ANT PWR HR NR NW "
    "OP QSO QTH RST RIG ANT PWR HR NR NW TU "
    "DR OM ES YL XYL HW CPY "
    "AGN RPT MSG NR TFC TU "
)


# ── N-gram model ──────────────────────────────────────────────────────────────

class _NgramModel:
    """
    Character-level N-gram model with Laplace (add-1) smoothing and linear
    interpolation across orders 1 … N.

    Weights λ favour the highest order that has evidence:
        λ = (0.10, 0.30, 0.60)  for unigram, bigram, trigram
    """

    _LAMBDAS = (0.10, 0.30, 0.60)

    def __init__(self, order: int = 3) -> None:
        self.order = order
        # _counts[context_str][char] = raw frequency
        self._counts: Dict[str, Counter] = defaultdict(Counter)
        self._vocab: Set[str] = set()

    def train(self, text: str) -> None:
        pad = "\x00" * (self.order - 1)
        s = pad + text
        for i in range(len(text)):
            char = s[i + self.order - 1]
            self._vocab.add(char)
            for k in range(1, self.order + 1):
                ctx = s[i + self.order - k : i + self.order - 1]
                self._counts[ctx][char] += 1

    def log_prob(self, char: str, context: str) -> float:
        """
        Interpolated log-probability log(Σ λₖ · Pₖ(char | context)).
        `context` is the (order-1) characters immediately preceding `char`.
        """
        v = max(1, len(self._vocab))
        lam = self._LAMBDAS
        p = 0.0

        # Unigram
        uni = self._counts.get("", Counter())
        p += lam[0] * (uni.get(char, 0) + 1) / (sum(uni.values()) + v)

        if self.order >= 2:
            bi_ctx = context[-1:] if context else ""
            bi = self._counts.get(bi_ctx, Counter())
            p += lam[1] * (bi.get(char, 0) + 1) / (sum(bi.values()) + v)

        if self.order >= 3:
            tri_ctx = context[-2:] if len(context) >= 2 else context
            tri = self._counts.get(tri_ctx, Counter())
            p += lam[2] * (tri.get(char, 0) + 1) / (sum(tri.values()) + v)

        return math.log(max(p, 1e-300))


# ── Corrector ─────────────────────────────────────────────────────────────────

class MorseCorrector:
    """
    Probabilistic post-processor for DSP-decoded Morse text.

    Pass 1 — [?] fill
        Each [?] token is replaced by the character that maximises
        P_forward(char | left_context) + P_backward(char | right_context).

    Pass 2 — confusion fix
        For each character in a Morse confusion set, swap it to the most
        probable partner only when that partner is ≥ CONF_THRESHOLD times
        more likely in the preceding trigram context.

    Both passes are pure-Python and run in < 1 ms per typical transmission.
    """

    # Only substitute a confused character when the alternative is this many
    # times more probable.  High value = conservative = fewer false positives.
    CONF_THRESHOLD: float = 15.0

    # Characters the DSP decoder can produce (excludes [?] itself).
    # Used as the candidate set when filling unknown tokens.
    _FILL_CHARS: Set[str] = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?'/-\"@=+$!"
    )

    def __init__(self, order: int = 3) -> None:
        self._fwd = _NgramModel(order)
        self._bwd = _NgramModel(order)
        self._log_thresh = math.log(self.CONF_THRESHOLD)
        self._build()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        self._fwd.train(_CORPUS)
        self._bwd.train(_CORPUS[::-1])

    # ── Public API ────────────────────────────────────────────────────────────

    def correct(self, text: str) -> str:
        """
        Apply both correction passes and return the cleaned text.
        Non-Morse status strings (e.g. "[Signal too weak …]") are returned
        unchanged — they are detected by the absence of [?] and the presence
        of '[' at position 0.
        """
        if text.startswith("[") and "[?]" not in text:
            return text
        text = self._fill_unknowns(text)
        text = self._fix_confusions(text)
        return text

    # ── Pass 1: fill [?] tokens ───────────────────────────────────────────────

    def _fill_unknowns(self, text: str) -> str:
        # Flatten text into a list of Optional[str] where None = [?] slot.
        tokens = re.split(r"(\[\?\])", text)
        flat: List[Optional[str]] = []
        for tok in tokens:
            if tok == "[?]":
                flat.append(None)
            else:
                flat.extend(list(tok))

        # Fill each None slot left-to-right (already-filled slots feed context
        # for subsequent slots, approximating a greedy Viterbi pass).
        order = self._fwd.order
        result: List[str] = []
        for i, ch in enumerate(flat):
            if ch is not None:
                result.append(ch)
                flat[i] = ch   # update so right-neighbour slots see it
            else:
                best = self._best_char(flat, i, order)
                result.append(best)
                flat[i] = best

        return "".join(result)

    def _best_char(
        self, flat: List[Optional[str]], pos: int, order: int
    ) -> str:
        """Return the highest-scoring candidate for the [?] at `pos`."""
        # Left context (forward model)
        left: List[str] = []
        for j in range(pos - 1, max(-1, pos - order), -1):
            if j < 0 or flat[j] is None:
                break
            left.insert(0, flat[j])
        left_ctx = "".join(left)[-(order - 1):]

        # Right context (backward model — collect chars to the right, reverse)
        right: List[str] = []
        for j in range(pos + 1, min(len(flat), pos + order)):
            if flat[j] is None:
                break
            right.append(flat[j])
        right_ctx = "".join(right[: order - 1])[::-1]

        best_char, best_score = next(iter(self._FILL_CHARS)), float("-inf")
        for c in self._FILL_CHARS:
            score = (
                self._fwd.log_prob(c, left_ctx)
                + self._bwd.log_prob(c, right_ctx)
            )
            if score > best_score:
                best_score = score
                best_char = c
        return best_char

    # ── Pass 2: confusion fixes ───────────────────────────────────────────────

    def _fix_confusions(self, text: str) -> str:
        """
        Walk character by character.  When a character belongs to a confusion
        set, score it against each partner using the forward model.  Swap only
        when the partner's log-probability exceeds the current character's by
        more than log(CONF_THRESHOLD).
        """
        chars = list(text)
        order = self._fwd.order

        for i, ch in enumerate(chars):
            if ch not in _CONFUSION_MAP:
                continue
            ctx = "".join(chars[max(0, i - (order - 1)) : i])
            current_lp = self._fwd.log_prob(ch, ctx)

            best_alt, best_alt_lp = ch, current_lp
            for alt in _CONFUSION_MAP[ch]:
                alt_lp = self._fwd.log_prob(alt, ctx)
                if alt_lp > best_alt_lp:
                    best_alt_lp = alt_lp
                    best_alt = alt

            if best_alt != ch and (best_alt_lp - current_lp) > self._log_thresh:
                chars[i] = best_alt

        return "".join(chars)


# ── Module-level singleton (optional convenience) ─────────────────────────────

_default: Optional[MorseCorrector] = None


def correct(text: str) -> str:
    """
    Module-level convenience wrapper.  Builds the shared MorseCorrector on
    first call (< 5 ms); subsequent calls are instant.
    """
    global _default
    if _default is None:
        _default = MorseCorrector()
    return _default.correct(text)
