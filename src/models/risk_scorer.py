"""
Dynamic Negative Word & Harm Scorer for Multilingual AI Safety.
Analyzes negative, toxic, adversarial, and high-risk terms across English,
Odia (ଓଡ଼ିଆ), Romanized Odia, Code-Mixed, and Hindi to dynamically scale the risk score.
Includes jailbreak pretext detection for fictional, hypothetical, and roleplay framing.
"""

import re
from typing import Any, Dict, List, Set, Tuple


class DynamicHarmScorer:
    """
    Computes a continuous risk score (0-100) and detects negative/adversarial
    linguistic tokens across English and Indian languages.
    Detects contextual intent concealment (e.g. 'writing a story about robbing a bank').
    """

    def __init__(self):
        # Category 1: Severe Harm & Violence (Weight: 55 points)
        self.severe_negative_terms = {
            # English
            "kill", "murder", "assassinate", "slaughter", "die", "death", "strangle",
            "stab", "shoot", "poison", "suicide", "torture", "weapon", "bomb", "explosive",
            "detonate", "grenade", "bioweapon", "cyanide", "massacre",
            # Odia native
            "ମାରିବା", "ହତ୍ୟା", "ମାରିଦେବା", "ବୋମା", "ବିଷ", "ବନ୍ଧୁକ", "ଆକ୍ରମଣ", "ଆତ୍ମହତ୍ୟା", "ମୃତ୍ୟୁ",
            # Romanized Odia & Code-Mixed
            "mariba", "hatya", "marideba", "boma", "bisha", "bandhuka", "akramana",
            "atmahathya", "mrutyu", "kill kariba", "murder kariba", "bomb baneiba",
            # Hindi
            "मारना", "हत्या", "मार डालना", "बम", "जहर", "हथियार", "हमला", "आत्महत्या", "मौत",
        }

        # Category 2: Illicit Acts, Crime & Robbery (Weight: 50 points)
        self.illicit_crime_terms = {
            # English
            "rob a bank", "rob", "robbery", "heist", "steal", "theft", "thief", "burglar",
            "burglary", "shoplift", "break into", "lockpick", "crack a safe", "pickpocket",
            "counterfeit", "kidnap", "extort", "blackmail",
            # Odia native
            "ଲୁଟ", "ଚୋରି", "ଚୋର", "ବ୍ୟାଙ୍କ ଲୁଟ", "ଡକାୟତି", "ହରଣ", "ସିନ୍ଧି ମାରିବା", "ବ୍ୟାଙ୍କ ଚୋରି",
            # Romanized Odia & Code-Mixed
            "luta", "bank luta", "chori", "chora", "dakaite", "rob kariba", "steal kariba",
            "bank rob", "bank chori",
            # Hindi
            "डकैती", "लूट", "बैंक लूटना", "चोरी", "चोर", "ताला तोड़ना", "अपहरण", "बैंक डकैती",
        }

        # Category 3: Malicious Cyber & Filter Evasion (Weight: 45 points)
        self.cyber_and_evasion_terms = {
            # English
            "bypass", "evade", "jailbreak", "circumvent", "exploit", "hack", "crack password",
            "steal credentials", "unauthorized access", "malware", "ransomware", "keylogger",
            "trojan", "ddos attack", "phishing", "backdoor",
            # Odia native
            "ବାଇପାସ୍", "ଏଡ଼ାଇବା", "ପାସୱାର୍ଡ ଚୋରି", "ଅନଧିକୃତ", "ହ୍ୟାକ୍",
            # Romanized Odia & Code-Mixed
            "bypass kariba", "edaiba", "filter evade", "password chori", "hack kariba",
            "unauthorized access", "credential steal", "jailbreak kariba",
            # Hindi
            "बायपास", "बचना", "पासवर्ड चोरी", "अनाधिकृत पहुंच", "हैक करना", "मैलवेयर",
        }

        # Category 4: Jailbreak / Concealment Wrappers (Roleplay, Fiction, Hypothetical)
        self.jailbreak_framing_wrappers = {
            # English
            "writing a story", "for a story", "fictional story", "in a novel", "movie script",
            "roleplay", "pretend that", "hypothetically", "hypothetical scenario", "fictional character",
            "in my book", "character is a thief", "acting as a criminal",
            # Odia native
            "ଗଳ୍ପ ଲେଖୁଛି", "କାହାଣୀ ପାଇଁ", "କାଳ୍ପନିକ", "ନାଟକ ପାଇଁ", "ଉପନ୍ୟାସ", "ଚରିତ୍ର ଏକ ଚୋର",
            # Romanized Odia & Code-Mixed
            "story lekhuchi", "kahani pain", "fictional story", "nataka pain", "roleplay re",
            "character thief",
            # Hindi
            "कहानी लिख रहा हूँ", "काल्पनिक कहानी", "उपन्यास के लिए", "नाटक के लिए", "काल्पनिक",
        }

        # Category 5: Hostile & Destructive Sentiment (Weight: 20 points)
        self.hostile_negative_terms = {
            # English
            "hate", "destroy", "damage", "attack", "illegal", "threat", "harm", "abuse",
            "corrupt", "danger", "ruin", "sabotage", "terror",
            # Odia native
            "ଘୃଣା", "ନଷ୍ଟ", "କ୍ଷତି", "ବିପଦ", "ଅପରାଧ", "ଅବୈଧ", "ଭୟଙ୍କର",
            # Romanized Odia & Code-Mixed
            "ghruna", "nasta", "khati", "bipada", "aparadha", "abaidha", "destroy kariba",
            # Hindi
            "नफरत", "नष्ट", "हानि", "खतरा", "अपराध", "अवैध", "बर्बाद",
        }

        # Category 6: Defensive Mitigators (Reduces score if legitimately defensive)
        self.defensive_mitigators = {
            # English
            "prevent", "safeguard", "protect", "defense", "security", "firewall", "antivirus",
            "mitigate", "avoid harm", "safety guidelines", "stop attack",
            # Odia native
            "ରୋକିବା", "ସୁରକ୍ଷା", "ପ୍ରତିରୋଧକ", "ନିରାପତ୍ତା", "ରକ୍ଷା",
            # Romanized Odia & Code-Mixed
            "rokiba", "surakshya", "prevent kariba", "protection", "security settings",
            # Hindi
            "रोकथाम", "सुरक्षा", "बचाव", "सुरक्षित",
        }

    def _extract_matches(self, text_lower: str, term_set: Set[str]) -> List[str]:
        """Find matching terms using regex boundaries for Latin or substring for Indic scripts."""
        matches = []
        for term in term_set:
            term_clean = term.lower().strip()
            if " " in term_clean or any(ord(c) > 127 for c in term_clean):
                if term_clean in text_lower:
                    matches.append(term_clean)
            else:
                pattern = rf"\b{re.escape(term_clean)}\b"
                if re.search(pattern, text_lower):
                    matches.append(term_clean)
        return matches

    def calculate_risk(self, text: str) -> Dict[str, Any]:
        """
        Calculates a dynamic risk score from 0.0 to 100.0 based on negative word intensity.
        Detects contextual concealment (e.g. fictional pretexts for criminal execution).
        """
        text_lower = text.lower().strip()

        severe_matches = self._extract_matches(text_lower, self.severe_negative_terms)
        crime_matches = self._extract_matches(text_lower, self.illicit_crime_terms)
        cyber_matches = self._extract_matches(text_lower, self.cyber_and_evasion_terms)
        framing_matches = self._extract_matches(text_lower, self.jailbreak_framing_wrappers)
        hostile_matches = self._extract_matches(text_lower, self.hostile_negative_terms)
        mitigator_matches = self._extract_matches(text_lower, self.defensive_mitigators)

        # Baseline score
        score = 10.0

        # Check for Adversarial Intent Concealment (Jailbreak Pretext)
        # Pairing a fictional / hypothetical wrapper with an actionable crime or violence request
        is_jailbreak_concealment = bool(framing_matches and (crime_matches or severe_matches or cyber_matches))

        if is_jailbreak_concealment:
            # Huge elevation: Fictional wrapper attempting to elicit illegal guidance
            score = 94.0 + (len(crime_matches) + len(severe_matches)) * 2.0
        elif severe_matches:
            # Severe violence / death / weapons immediately starts in high risk tier
            score = 88.0 + (len(severe_matches) - 1) * 6.0
        else:
            # Standard increments
            score += len(crime_matches) * 50.0
            score += len(cyber_matches) * 45.0
            score += len(hostile_matches) * 20.0

            # Defensive mitigations (only if not severe harm or crime)
            if mitigator_matches and not severe_matches and not crime_matches:
                score -= len(mitigator_matches) * 35.0

        # Clamp between 5.0 and 100.0
        score = max(5.0, min(100.0, score))
        score = round(score, 1)

        # Map to Risk Level category
        if score >= 70.0:
            level = "High"
        elif score >= 35.0:
            level = "Medium"
        else:
            level = "Low"

        all_detected = severe_matches + crime_matches + cyber_matches + hostile_matches
        if is_jailbreak_concealment:
            all_detected.extend([f"pretext: '{f}'" for f in framing_matches])

        return {
            "risk_score": score,
            "risk_level": level,
            "detected_negative_words": list(set(all_detected)),
            "detected_mitigators": list(set(mitigator_matches)),
            "is_jailbreak_concealment": is_jailbreak_concealment,
            "jailbreak_pretexts": framing_matches,
            "severe_word_count": len(severe_matches),
            "crime_word_count": len(crime_matches),
            "threat_word_count": len(cyber_matches),
        }
