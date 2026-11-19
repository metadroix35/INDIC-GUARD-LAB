"""
Dynamic Negative Word & Harm Scorer for Multilingual AI Safety.
Analyzes negative, toxic, adversarial, and high-risk terms across English,
Odia (ଓଡ଼ିଆ), Romanized Odia, Code-Mixed, and Hindi to dynamically scale the risk score.
"""

import re
from typing import Any, Dict, List, Set, Tuple


class DynamicHarmScorer:
    """
    Computes a continuous risk score (0-100) and detects negative/adversarial
    linguistic tokens across English and Indian languages.
    """

    def __init__(self):
        # Category 1: Severe Harm & Violence (Weight: 50 points each)
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

        # Category 2: Malicious Cyber & Evasion (Weight: 40 points each)
        self.cyber_and_evasion_terms = {
            # English
            "bypass", "evade", "jailbreak", "circumvent", "exploit", "hack", "crack password",
            "steal credentials", "unauthorized access", "malware", "ransomware", "keylogger",
            "trojan", "ddos attack", "phishing", "backdoor",
            # Odia native
            "ବାଇପାସ୍", "ଏଡ଼ାଇବା", "ଚୋରି", "ପାସୱାର୍ଡ ଚୋରି", "ଅନଧିକୃତ", "ହ୍ୟାକ୍",
            # Romanized Odia & Code-Mixed
            "bypass kariba", "edaiba", "filter evade", "password chori", "hack kariba",
            "unauthorized access", "credential steal", "jailbreak kariba",
            # Hindi
            "बायपास", "बचना", "पासवर्ड चोरी", "अनाधिकृत पहुंच", "हैक करना", "मैलवेयर",
        }

        # Category 3: General Negative & Hostile Sentiment (Weight: 20 points each)
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

        # Category 4: Defensive Mitigators (Reduces negative score if legitimately defensive)
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
        Returns:
            - numerical risk_score (0-100)
            - categorical risk_level ('Low', 'Medium', 'High')
            - detected negative words
            - defensive mitigators detected
        """
        text_lower = text.lower().strip()

        severe_matches = self._extract_matches(text_lower, self.severe_negative_terms)
        cyber_matches = self._extract_matches(text_lower, self.cyber_and_evasion_terms)
        hostile_matches = self._extract_matches(text_lower, self.hostile_negative_terms)
        mitigator_matches = self._extract_matches(text_lower, self.defensive_mitigators)

        # Baseline score
        score = 10.0

        # Increment for severe violence/harm (huge jump)
        score += len(severe_matches) * 55.0

        # Increment for cyber / evasion attempts
        score += len(cyber_matches) * 45.0

        # Increment for hostile/destructive terms
        score += len(hostile_matches) * 20.0

        # Defensive mitigations (e.g. "how do firewalls prevent DDoS attacks")
        if mitigator_matches:
            # If the query is explicitly asking to prevent/safeguard, decrease score
            if not severe_matches:  # Do not mitigate murder/kill queries
                score -= len(mitigator_matches) * 35.0

        # Clamp between 0.0 and 100.0
        score = max(5.0, min(100.0, score))
        score = round(score, 1)

        # Map to Risk Level category
        if score >= 70.0:
            level = "High"
        elif score >= 35.0:
            level = "Medium"
        else:
            level = "Low"

        all_detected = severe_matches + cyber_matches + hostile_matches

        return {
            "risk_score": score,
            "risk_level": level,
            "detected_negative_words": list(set(all_detected)),
            "detected_mitigators": list(set(mitigator_matches)),
            "severe_word_count": len(severe_matches),
            "threat_word_count": len(cyber_matches),
            "hostile_word_count": len(hostile_matches),
        }
