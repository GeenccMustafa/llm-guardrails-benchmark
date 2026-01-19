"""Implementations of different guardrail frameworks."""

from typing import Dict, Any, List
from abc import ABC, abstractmethod
import time
from loguru import logger
import re

# LLM Guard
try:
    from llm_guard.input_scanners import (
        Toxicity as LLMGuardToxicity,
        PromptInjection,
        TokenLimit,
    )
    from llm_guard.output_scanners import (
        Toxicity as OutputToxicity,
        NoRefusal,
        Sensitive
    )
    from llm_guard.vault import Vault
    LLM_GUARD_AVAILABLE = True
except ImportError:
    logger.warning("LLM Guard not available")
    LLM_GUARD_AVAILABLE = False

# Presidio
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    logger.warning("Presidio not available")
    PRESIDIO_AVAILABLE = False


class BaseGuardrail(ABC):
    """Base class for all guardrails."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def check_input(self, text: str) -> Dict[str, Any]:
        """Check if input passes guardrail."""
        pass
    
    @abstractmethod
    def check_output(self, text: str) -> Dict[str, Any]:
        """Check if output passes guardrail."""
        pass


class SimpleRuleBasedGuardrail(BaseGuardrail):
    """Enhanced rule-based guardrail with more patterns."""
    
    def __init__(self):
        super().__init__("SimpleRuleBased")
        
        # Enhanced toxic patterns
        self.toxic_patterns = [
            r'\b(stupid|idiot|dumb|moron|worthless|hate|loser|garbage)\b',
            r'\b(kill|murder|die|death|suicide)\b',
            r'\b(fuck|shit|damn|hell|ass|bitch)\b',
            r'\b(racist|sexist|nazi|terrorism)\b',
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        
        # Enhanced jailbreak patterns
        self.jailbreak_patterns = [
            r'ignore (all )?(previous|prior|above) (instructions|rules|directions)',
            r'(DAN|Developer) mode',
            r'you are now',
            r'forget (all )?(your|previous)',
            r'disregard (all )?(previous|prior|above)',
            r'act as if',
            r'pretend (you are|to be)',
            r'roleplay as',
            r'new instructions:',
            r'system prompt:',
        ]
        
        # Code injection patterns
        self.code_patterns = [
            r'(import\s+os|from\s+os\s+import)',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'system\s*\(',
            r'rm\s+-rf',
            r'DROP\s+TABLE',
        ]
        
        logger.info("SimpleRuleBasedGuardrail initialized with enhanced patterns")
    
    def _check_text(self, text: str) -> Dict[str, Any]:
        """Check text against patterns."""
        text_lower = text.lower()
        reasons = []
        
        # Check toxic patterns
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                reasons.append(f"Toxic language detected")
                break
        
        # Check PII
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                reasons.append(f"PII detected: {pii_type}")
        
        # Check jailbreak
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, text_lower):
                reasons.append(f"Jailbreak attempt detected")
                break
        
        # Check code injection
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                reasons.append(f"Code injection attempt detected")
                break
        
        return {
            "blocked": len(reasons) > 0,
            "reason": "; ".join(reasons) if reasons else None,
        }
    
    def check_input(self, text: str) -> Dict[str, Any]:
        """Check input with rule-based patterns."""
        start_time = time.time()
        result = self._check_text(text)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            **result,
            "time_ms": elapsed_ms,
            "details": None
        }
    
    def check_output(self, text: str) -> Dict[str, Any]:
        """Check output with rule-based patterns."""
        start_time = time.time()
        result = self._check_text(text)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            **result,
            "time_ms": elapsed_ms,
            "details": None
        }


class LLMGuardImplementation(BaseGuardrail):
    """LLM Guard implementation with vault support."""
    
    def __init__(self):
        super().__init__("LLMGuard")
        
        if not LLM_GUARD_AVAILABLE:
            self.input_scanners = []
            self.output_scanners = []
            return
        
        try:
            # Initialize vault for anonymization
            self.vault = Vault()
            
            # Initialize scanners (removed Anonymize that needs vault config)
            self.input_scanners = [
                LLMGuardToxicity(threshold=0.5),
                PromptInjection(threshold=0.5),
                TokenLimit(limit=4096, encoding_name="cl100k_base"),
            ]
            
            self.output_scanners = [
                OutputToxicity(threshold=0.5),
                Sensitive(entity_types=["EMAIL_ADDRESS", "PHONE_NUMBER"]),
            ]
            
            logger.info("LLMGuard initialized successfully with vault")
        except Exception as e:
            logger.error(f"Error initializing LLMGuard: {e}")
            self.input_scanners = []
            self.output_scanners = []
    
    def check_input(self, text: str) -> Dict[str, Any]:
        """Check input with LLM Guard."""
        if not self.input_scanners:
            return {"blocked": False, "reason": "Scanners not initialized", "time_ms": 0, "details": None}
        
        start_time = time.time()
        
        try:
            sanitized_prompt = text
            risks = []
            
            for scanner in self.input_scanners:
                sanitized_prompt, is_valid, risk_score = scanner.scan(sanitized_prompt)
                if not is_valid:
                    risks.append(f"{scanner.__class__.__name__}: {risk_score:.2f}")
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                "blocked": len(risks) > 0,
                "reason": "; ".join(risks) if risks else None,
                "time_ms": elapsed_ms,
                "details": {"sanitized": sanitized_prompt[:100]}
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"LLMGuard input error: {e}")
            return {
                "blocked": False,
                "reason": f"Error: {str(e)}",
                "time_ms": elapsed_ms,
                "details": None
            }
    
    def check_output(self, text: str) -> Dict[str, Any]:
        """Check output with LLM Guard."""
        if not self.output_scanners:
            return {"blocked": False, "reason": "Scanners not initialized", "time_ms": 0, "details": None}
        
        start_time = time.time()
        
        try:
            sanitized_output = text
            risks = []
            
            for scanner in self.output_scanners:
                sanitized_output, is_valid, risk_score = scanner.scan(sanitized_output)
                if not is_valid:
                    risks.append(f"{scanner.__class__.__name__}: {risk_score:.2f}")
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                "blocked": len(risks) > 0,
                "reason": "; ".join(risks) if risks else None,
                "time_ms": elapsed_ms,
                "details": {"sanitized": sanitized_output[:100]}
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"LLMGuard output error: {e}")
            return {
                "blocked": False,
                "reason": f"Error: {str(e)}",
                "time_ms": elapsed_ms,
                "details": None
            }


class PresidioImplementation(BaseGuardrail):
    """Presidio PII detection implementation."""
    
    def __init__(self):
        super().__init__("Presidio")
        
        if not PRESIDIO_AVAILABLE:
            self.analyzer = None
            self.anonymizer = None
            return
        
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            logger.info("Presidio initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Presidio: {e}")
            self.analyzer = None
            self.anonymizer = None
    
    def _check_pii(self, text: str) -> Dict[str, Any]:
        """Check for PII in text."""
        if not self.analyzer:
            return {"blocked": False, "reason": "Analyzer not initialized", "time_ms": 0}
        
        start_time = time.time()
        
        try:
            results = self.analyzer.analyze(text=text, language='en')
            elapsed_ms = (time.time() - start_time) * 1000
            
            if results:
                entities = [f"{r.entity_type}({r.score:.2f})" for r in results]
                return {
                    "blocked": True,
                    "reason": f"PII detected: {', '.join(entities)}",
                    "time_ms": elapsed_ms,
                    "details": {"entities": entities}
                }
            else:
                return {
                    "blocked": False,
                    "reason": None,
                    "time_ms": elapsed_ms,
                    "details": None
                }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Presidio error: {e}")
            return {
                "blocked": False,
                "reason": f"Error: {str(e)}",
                "time_ms": elapsed_ms,
                "details": None
            }
    
    def check_input(self, text: str) -> Dict[str, Any]:
        """Check input for PII."""
        return self._check_pii(text)
    
    def check_output(self, text: str) -> Dict[str, Any]:
        """Check output for PII."""
        return self._check_pii(text)


class CombinedGuardrail(BaseGuardrail):
    """Combined guardrail using all three in sequence."""
    
    def __init__(self):
        super().__init__("Combined")
        
        self.guardrails = []
        
        # Add all available guardrails
        try:
            self.guardrails.append(SimpleRuleBasedGuardrail())
        except Exception as e:
            logger.warning(f"Could not add SimpleRuleBased: {e}")
        
        if LLM_GUARD_AVAILABLE:
            try:
                self.guardrails.append(LLMGuardImplementation())
            except Exception as e:
                logger.warning(f"Could not add LLMGuard: {e}")
        
        if PRESIDIO_AVAILABLE:
            try:
                self.guardrails.append(PresidioImplementation())
            except Exception as e:
                logger.warning(f"Could not add Presidio: {e}")
        
        logger.info(f"CombinedGuardrail initialized with {len(self.guardrails)} guardrails")
    
    def check_input(self, text: str) -> Dict[str, Any]:
        """Check input with all guardrails."""
        start_time = time.time()
        
        all_reasons = []
        total_blocked = False
        
        for guardrail in self.guardrails:
            result = guardrail.check_input(text)
            if result['blocked']:
                total_blocked = True
                all_reasons.append(f"{guardrail.name}: {result['reason']}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "blocked": total_blocked,
            "reason": "; ".join(all_reasons) if all_reasons else None,
            "time_ms": elapsed_ms,
            "details": {"guardrails_checked": len(self.guardrails)}
        }
    
    def check_output(self, text: str) -> Dict[str, Any]:
        """Check output with all guardrails."""
        start_time = time.time()
        
        all_reasons = []
        total_blocked = False
        
        for guardrail in self.guardrails:
            result = guardrail.check_output(text)
            if result['blocked']:
                total_blocked = True
                all_reasons.append(f"{guardrail.name}: {result['reason']}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "blocked": total_blocked,
            "reason": "; ".join(all_reasons) if all_reasons else None,
            "time_ms": elapsed_ms,
            "details": {"guardrails_checked": len(self.guardrails)}
        }


# Factory function
def get_all_guardrails() -> List[BaseGuardrail]:
    """Get all available guardrail implementations."""
    guardrails = []
    
    # Add individual guardrails
    try:
        guardrails.append(SimpleRuleBasedGuardrail())
    except Exception as e:
        logger.warning(f"Could not initialize SimpleRuleBased: {e}")
    
    if LLM_GUARD_AVAILABLE:
        try:
            guardrails.append(LLMGuardImplementation())
        except Exception as e:
            logger.warning(f"Could not initialize LLMGuard: {e}")
    
    if PRESIDIO_AVAILABLE:
        try:
            guardrails.append(PresidioImplementation())
        except Exception as e:
            logger.warning(f"Could not initialize Presidio: {e}")
    
    # Add combined guardrail
    try:
        guardrails.append(CombinedGuardrail())
    except Exception as e:
        logger.warning(f"Could not initialize CombinedGuardrail: {e}")
    
    return guardrails