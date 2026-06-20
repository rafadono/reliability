"""
Vendor-Agnostic LLM Service Adapter.
Supports Mock, Google Gemini, OpenAI, and Ollama (local) engines via HTTP requests.
Provides unified structure for RCM suggestions, RCA suggestions, and general Copilot chats.
"""

import json
import logging
import requests
from typing import Dict, List, Any
import src.reliability_analysis.utils.config as config

logger = logging.getLogger(__name__)


class LlmService:
    """Unified service adapter to handle multi-vendor LLM providers."""

    @classmethod
    def _call_api(cls, prompt: str, system_instruction: str = "") -> str:
        """Internal helper to make HTTP requests to the configured provider."""
        provider = config.LLM_PROVIDER
        model = config.LLM_MODEL

        if provider == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured")
            
            headers = {
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model or "gpt-4o",
                "messages": messages,
                "response_format": {"type": "json_object"}
            }
            try:
                res = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"OpenAI API call failed: {str(e)}")
                raise

        elif provider == "gemini":
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured")
            
            target_model = model or "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={config.GEMINI_API_KEY}"
            
            headers = {"Content-Type": "application/json"}
            
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                # Extract text from candidate response structure
                candidates = res.json().get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"]
                raise ValueError(f"Empty response from Gemini: {res.text}")
            except Exception as e:
                logger.error(f"Gemini API call failed: {str(e)}")
                raise

        elif provider == "ollama":
            url = f"{config.OLLAMA_BASE_URL}/api/chat"
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model or "llama3",
                "messages": messages,
                "stream": False,
                "format": "json"
            }
            try:
                res = requests.post(url, json=payload, timeout=45)
                res.raise_for_status()
                return res.json()["message"]["content"]
            except Exception as e:
                logger.error(f"Ollama local API call failed: {str(e)}")
                raise

        else:
            # Fallback mock/heuristic response
            raise ValueError(f"Provider '{provider}' is not supported or set to mock")

    @classmethod
    def get_rcm_suggestions(cls, equipment: str, technical_comments: List[str]) -> List[Dict[str, Any]]:
        """
        Generates 7-question RCM sheets complying with SAE JA1011.
        Uses LLM if configured; otherwise falls back to static dictionary mapping.
        """
        provider = config.LLM_PROVIDER
        if provider == "mock" or not provider:
            # Use deterministic fallback
            return cls._get_mock_rcm(equipment, technical_comments)

        comments_str = "\n".join([f"- {c}" for c in technical_comments[:15]])
        system_instruction = (
            "Eres un Ingeniero Especialista en RCM bajo la norma SAE JA1011/12. "
            "Debes responder UNICAMENTE en formato JSON con la siguiente estructura de lista: "
            "[{\"function\": \"...\", \"functional_failure\": \"...\", \"mode\": \"...\", \"effect\": \"...\", \"consequence\": \"...\", \"proactive_task\": \"...\", \"alternative_action\": \"...\"}]"
        )
        prompt = (
            f"Analiza el equipo '{equipment}' basándote en los siguientes comentarios del historial técnico:\n"
            f"{comments_str}\n\n"
            f"Genera al menos 2 fichas RCM con fallos funcionales y modos de falla representativos."
        )

        try:
            response_text = cls._call_api(prompt, system_instruction)
            return json.loads(response_text)
        except Exception as e:
            logger.warn(f"LLM RCM suggestion failed: {str(e)}. Falling back to mock heuristics.")
            return cls._get_mock_rcm(equipment, technical_comments)

    @classmethod
    def get_rca_suggestions(cls, equipment: str, technical_comments: List[str]) -> Dict[str, Any]:
        """
        Generates 5 Whys and Ishikawa diagrams complying with IEC 62740.
        Uses LLM if configured; otherwise falls back to static dictionary mapping.
        """
        provider = config.LLM_PROVIDER
        if provider == "mock" or not provider:
            return cls._get_mock_rca(equipment, technical_comments)

        comments_str = "\n".join([f"- {c}" for c in technical_comments[:15]])
        system_instruction = (
            "Eres un experto en Análisis de Causa Raíz (RCA) según la norma IEC 62740. "
            "Debes responder UNICAMENTE en formato JSON con la siguiente estructura: "
            "{\n"
            "  \"five_whys\": [\n"
            "    {\"question\": \"...\", \"answer\": \"...\"}\n"
            "  ],\n"
            "  \"ishikawa\": {\n"
            "    \"machinery\": [\"...\", \"...\"],\n"
            "    \"method\": [\"...\", \"...\"],\n"
            "    \"workforce\": [\"...\", \"...\"],\n"
            "    \"environment\": [\"...\", \"...\"]\n"
            "  }\n"
            "}"
        )
        prompt = (
            f"Analiza la falla del equipo '{equipment}' basándote en los comentarios técnicos:\n"
            f"{comments_str}\n\n"
            f"Genera la cadena de 5 Porqués y clasifica las causas en el diagrama de Ishikawa."
        )

        try:
            response_text = cls._call_api(prompt, system_instruction)
            return json.loads(response_text)
        except Exception as e:
            logger.warn(f"LLM RCA suggestion failed: {str(e)}. Falling back to mock heuristics.")
            return cls._get_mock_rca(equipment, technical_comments)

    @classmethod
    def _get_mock_rcm(cls, equipment: str, technical_comments: List[str]) -> List[Dict[str, Any]]:
        """Deterministic mock fallback for RCM."""
        equipment_function = f"Proporcionar bombeo y flujo regulado de fluido de proceso para el sistema de {equipment} de acuerdo a las especificaciones de caudal nominal y presión de descarga."
        functional_failure = "Pérdida total de bombeo de fluido (caudal cero)."
        
        comments_lower = [c.lower() for c in technical_comments]
        has_seals = any("sello" in c or "fuga" in c or "leak" in c for c in comments_lower)
        has_bearings = any("rodamiento" in c or "bearing" in c or "vibra" in c for c in comments_lower)

        suggestions = [
            {
                "function": equipment_function,
                "functional_failure": functional_failure,
                "mode": "Desgaste y rotura del rodamiento principal",
                "effect": "Vibración excesiva, incremento anormal de temperatura, ruido fuerte y posterior atascamiento de rotor.",
                "consequence": "Parada imprevista de planta, pérdida de disponibilidad y costo elevado de repuestos.",
                "proactive_task": "Medición y análisis espectral de vibraciones quincenal; engrase según horas de servicio.",
                "alternative_action": "Operar a la rotura con repuesto disponible en almacén."
            },
            {
                "function": equipment_function,
                "functional_failure": "Pérdida parcial de flujo por fugas visibles",
                "mode": "Degradación del sello mecánico / fugas",
                "effect": "Pérdida gradual de fluido de proceso, baja de presión del sistema y contaminación del área circundante.",
                "consequence": "Impacto menor en producción, pero alto riesgo medioambiental y de seguridad.",
                "proactive_task": "Inspección visual diaria y medición de presión de barrera; cambio de sello programado anual.",
                "alternative_action": "Reemplazo inmediato por procedimiento de control ambiental."
            }
        ]

        if has_bearings and not has_seals:
            return [suggestions[0]]
        elif has_seals and not has_bearings:
            return [suggestions[1]]
        return suggestions

    @classmethod
    def _get_mock_rca(cls, equipment: str, technical_comments: List[str]) -> Dict[str, Any]:
        """Deterministic mock fallback for RCA."""
        return {
            "five_whys": [
                {"question": f"¿Por qué falló el rodamiento del {equipment}?", "answer": "Porque se produjo un sobrecalentamiento extremo y pérdida de lubricación en la pista."},
                {"question": "¿Por qué hubo pérdida de lubricación?", "answer": "Porque la grasa no ingresó al rodamiento durante la ronda quincenal."},
                {"question": "¿Por qué no ingresó la grasa?", "answer": "Porque la grasera o boquilla de inyección estaba completamente obstruida por suciedad endurecida."},
                {"question": "¿Por qué estaba obstruida la boquilla?", "answer": "Porque no se realiza una limpieza previa de los puntos de engrase en las rondas de mantenimiento."},
                {"question": "¿Por qué se omitió esa limpieza?", "answer": "Porque el procedimiento estándar de lubricación no especifica formalmente la limpieza de las graseras antes de bombear."}
            ],
            "ishikawa": {
                "machinery": [
                    "Boquilla de engrase (grasera) tapada u obstruida",
                    "Rodamiento con holgura excesiva por fin de vida útil"
                ],
                "method": [
                    "Procedimiento estándar de lubricación no detalla limpieza previa de boquillas",
                    "Rutas de lubricación sin control de torque o volumen inyectado"
                ],
                "workforce": [
                    "Omisión de la limpieza en la boquilla antes del bombeo de grasa",
                    "Falta de capacitación técnica sobre mecanismos de falla en rodamientos"
                ],
                "environment": [
                    "Exposición a polvo abrasivo y humedad en el área operacional",
                    "Altas temperaturas de operación que aceleran la degradación de la grasa"
                ]
            }
        }
