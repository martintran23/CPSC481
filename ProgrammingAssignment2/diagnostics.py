import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional


class Diagnostics:
    """
    Diagnose TB/Cancer/Bronchitis by asking an LLM to compute posterior
    probabilities in a given Bayesian network (no manual probability math).
    """

    def __init__(self):
        # Load environment variables from a local .env file (if present).
        # This keeps API keys out of the codebase.
        self._load_local_env()

        # Initialize Gemini config from environment variables.
        # Do NOT hardcode API keys in code.
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

        # Cache results so repeated GUI/test calls are fast.
        self._cache = {}

        # Optional local fallback (used when Gemini can't be called).
        self._local_tokenizer = None
        self._local_model = None
        self._local_model_id = os.getenv("LOCAL_LLM_MODEL")
        self._local_max_new_tokens = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "128"))

    @staticmethod
    def _load_local_env() -> None:
        """Minimal .env loader to avoid adding extra dependencies."""
        # Prefer repo-root .env (CPSC481/.env), but also support a local .env
        # inside ProgrammingAssignment2/ for convenience.
        this_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(this_dir, ".."))
        candidate_paths = [
            os.path.join(repo_root, ".env"),
            os.path.join(this_dir, ".env"),
        ]
        dotenv_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not dotenv_path:
            return

        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:  # type: ignore[arg-type]
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    # Only set if not already present in the process environment,
                    # or if it is present but blank (common when users started
                    # a process with an empty OPENAI_API_KEY).
                    if key and (key not in os.environ or not os.environ.get(key, "")):
                        os.environ[key] = value
        except Exception:
            # Silently ignore .env loading issues; code will fall back to existing env vars.
            return

    def _build_prompt(
        self,
        visit_to_asia: str,
        smoking: str,
        xray_result: str,
        dyspnea: str,
    ) -> str:
        # Ensure the prompt contains the full Bayesian network definition,
        # including CPTs and the observed evidence.
        vta = (visit_to_asia or "").strip().lower()
        smk = (smoking or "").strip().lower()
        xra = (xray_result or "").strip().lower()
        dyp = (dyspnea or "").strip().lower()

        evidence_lines: List[str] = []
        if vta in {"yes", "no"}:
            evidence_lines.append(f"- VisitToAsia = {visit_to_asia}")
        if smk in {"yes", "no"}:
            evidence_lines.append(f"- Smoking = {smoking}")
        if xra in {"positive", "negative"}:
            evidence_lines.append(f"- XRay = {xray_result}")
        if dyp in {"yes", "no"}:
            evidence_lines.append(f"- Dyspnea = {dyspnea}")

        if not evidence_lines:
            evidence_block = "- (No evidence provided)"
        else:
            evidence_block = "\n".join(evidence_lines)

        return (
            "You are given the following Bayesian network for probabilistic diagnosis.\n\n"
            "Variables:\n"
            "- VisitToAsia (yes/no)\n"
            "- Smoking (yes/no)\n"
            "- Tuberculosis (TB)\n"
            "- Lung Cancer (Cancer)\n"
            "- Bronchitis\n"
            "- Either (TB OR Cancer)\n"
            "- XRay\n"
            "- Dyspnea\n\n"
            "Probabilities:\n"
            "P(VisitToAsia = yes) = 0.01\n"
            "P(Smoking = yes) = 0.5\n\n"
            "P(TB = yes | VisitToAsia):\n"
            "- yes: 0.05\n"
            "- no: 0.01\n\n"
            "P(Cancer = yes | Smoking):\n"
            "- yes: 0.1\n"
            "- no: 0.01\n\n"
            "P(Bronchitis = yes | Smoking):\n"
            "- yes: 0.6\n"
            "- no: 0.3\n\n"
            "Either = TB OR Cancer (deterministic OR):\n"
            "P(Either = yes | TB, Cancer):\n"
            "- TB=yes, Cancer=yes: 1.0\n"
            "- TB=yes, Cancer=no: 1.0\n"
            "- TB=no, Cancer=yes: 1.0\n"
            "- TB=no, Cancer=no: 0.0\n\n"
            "P(XRay = positive | Either):\n"
            "- yes: 0.99\n"
            "- no: 0.05\n\n"
            "P(Dyspnea = yes | Either, Bronchitis):\n"
            "- Either=yes & Bronchitis=yes: 0.9\n"
            "- Either=yes & Bronchitis=no: 0.7\n"
            "- Either=no & Bronchitis=yes: 0.8\n"
            "- Either=no & Bronchitis=no: 0.1\n\n"
            "Observed evidence:\n"
            f"{evidence_block}\n\n"
            "Interpretation rules:\n"
            "- Any symptom variable NOT listed in the observed evidence must be treated as UNOBSERVED.\n"
            "- For UNOBSERVED variables, compute posteriors by summing over both yes/no (and positive/negative) possibilities.\n\n"
            "Bayesian network factorization (exact):\n"
            "P(VisitToAsia) * P(Smoking) * P(TB|VisitToAsia) * P(Cancer|Smoking) * P(Bronchitis|Smoking) * P(Either|TB,Cancer) * P(XRay|Either) * P(Dyspnea|Either,Bronchitis)\n\n"
            "Task:\n"
            "Compute the posterior probabilities given the observed evidence for:\n"
            "- TB\n"
            "- Cancer\n"
            "- Bronchitis\n\n"
            "Compute each posterior as: P(D=yes | evidence), where D is one of {TB, Cancer, Bronchitis}.\n"
            "Then choose the single disease with the highest posterior probability among TB, Cancer, and Bronchitis.\n\n"
            "Return ONLY the following JSON object (no extra text, no markdown):\n"
            "{\n"
            '  "disease": "TB",\n'
            "  \"probability\": 0.0\n"
            "}\n"
            "Constraints:\n"
            "- \"disease\" must be exactly one of: \"TB\", \"Cancer\", \"Bronchitis\"\n"
            "- \"probability\" must be the posterior probability of that chosen disease and must be a number between 0 and 1.\n"
            "- Round \"probability\" to exactly 3 decimal places (e.g., 0.255).\n"
            "- IMPORTANT: Do NOT reuse or copy the placeholder values above. You must compute the posterior probability from the Bayesian network and evidence provided, then replace the JSON fields with the computed results.\n"
        )

    @staticmethod
    def _normalize_yes_no(value: str) -> str:
        v = (value or "").strip().lower()
        if v in {"yes", "y", "true", "1", "present"}:
            return "yes"
        if v in {"no", "n", "false", "0", "absent"}:
            return "no"
        # If unknown, pass through unchanged; the LLM will still see evidence text.
        return value

    @staticmethod
    def _normalize_xray(value: str) -> str:
        v = (value or "").strip().lower()
        if v in {"positive", "abnormal", "abn", "true", "1"}:
            return "positive"
        if v in {"negative", "normal", "nor", "false", "0"}:
            return "negative"
        return value

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini `generateContent` REST API and return the generated text.
        """
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        )

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                # Strongly encourage JSON output when supported by the model.
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_text = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # Include response body for easier debugging.
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Gemini HTTPError: {e.code} {detail}") from e

        data = json.loads(resp_text)
        # Gemini response structure: candidates[0].content.parts[0].text
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {resp_text}")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            raise RuntimeError(f"Gemini returned no parts: {resp_text}")

        text = parts[0].get("text")
        if text is None:
            raise RuntimeError(f"Gemini returned no text: {resp_text}")

        return text

    def diagnose(
        self,
        visit_to_asia: str,
        smoking: str,
        xray_result: str,
        dyspnea: str,
    ) -> List:
        """
        Returns [disease_name, probability] where:
          - disease_name is one of "TB", "Cancer", "Bronchitis"
          - probability is a float between 0 and 1
        """
        # Normalize inputs to match evidence labels in the Bayesian network.
        visit_to_asia = self._normalize_yes_no(visit_to_asia)
        smoking = self._normalize_yes_no(smoking)
        xray_result = self._normalize_xray(xray_result)
        dyspnea = self._normalize_yes_no(dyspnea)

        cache_key = (visit_to_asia, smoking, xray_result, dyspnea)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        if not self.gemini_api_key:
            print("Error: GEMINI_API_KEY is not set.")
            return self._diagnose_with_local(
                visit_to_asia=visit_to_asia,
                smoking=smoking,
                xray_result=xray_result,
                dyspnea=dyspnea,
                cache_key=cache_key,
            )

        # Build the prompt with full BN structure + CPTs + observed evidence.
        prompt = self._build_prompt(
            visit_to_asia=visit_to_asia,
            smoking=smoking,
            xray_result=xray_result,
            dyspnea=dyspnea,
        )

        try:
            # Debugging requirement: print the full prompt sent to the LLM.
            print(prompt)

            # Call Gemini to compute posteriors via Bayesian-network reasoning.
            raw_response = self._call_gemini(prompt)
        except Exception as e:
            err = str(e)
            print(f"Error calling Gemini API: {err}")

            # If the API itself is disabled for the Google Cloud project,
            # falling back to a local model will produce wrong probabilities.
            # Surface a clear instruction instead.
            if "service_disabled" in err.lower() or "SERVICE_DISABLED" in err:
                print(
                    "Gemini failed because the Generative Language API is disabled "
                    "for the Google Cloud project tied to your API key.\n"
                    "Fix: enable generativelanguage.googleapis.com (Generative Language API) "
                    "in the Google Cloud Console for that project, then retry."
                )
                return ["TB", 0.0]

            # For other Gemini failures, fall back to local inference.
            return self._diagnose_with_local(
                visit_to_asia=visit_to_asia,
                smoking=smoking,
                xray_result=xray_result,
                dyspnea=dyspnea,
                cache_key=cache_key,
                prompt=prompt,
            )

        # Debugging requirement: print raw LLM response before parsing.
        print(raw_response)

        # Parse JSON safely using Python's json module.
        try:
            # Prefer strict JSON parsing first.
            raw = raw_response.strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # If the model returns extra text, extract the first JSON object.
                match = re.search(r"\{[\s\S]*?\}", raw, flags=re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    # Last-resort extraction for outputs that mention the keys but
                    # do not produce a full JSON object.
                    d_match = re.search(
                        r"\"disease\"\\s*:\\s*\"(TB|Cancer|Bronchitis)\"",
                        raw,
                        flags=re.IGNORECASE,
                    )
                    p_match = re.search(
                        r"\"probability\"\\s*:\\s*([01](?:\\.\\d+)?)",
                        raw,
                        flags=re.IGNORECASE,
                    )
                    if not d_match or not p_match:
                        raise

                    disease_extracted = d_match.group(1)
                    probability_extracted = float(p_match.group(1))
                    result = [disease_extracted, probability_extracted]
                    self._cache[cache_key] = tuple(result)
                    return result

            disease = parsed.get("disease")
            probability = parsed.get("probability")

            if disease not in {"TB", "Cancer", "Bronchitis"}:
                raise ValueError("Invalid disease returned by LLM.")

            prob_float = float(probability)
            if not (0.0 <= prob_float <= 1.0):
                raise ValueError("Probability out of range.")

            # Match Project 1's rounding-to-3-decimals behavior.
            result = [disease, round(prob_float, 3)]
            self._cache[cache_key] = tuple(result)
            return result
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return ["TB", 0.0]

    def _diagnose_with_local(
        self,
        visit_to_asia: str,
        smoking: str,
        xray_result: str,
        dyspnea: str,
        cache_key,
        prompt: Optional[str] = None,
    ) -> List:
        """
        Local fallback used when the Gemini API cannot be called.
        """
        if not self._local_model_id:
            print("Error: LOCAL_LLM_MODEL is not set for local fallback.")
            return ["TB", 0.0]

        if prompt is None:
            prompt = self._build_prompt(
                visit_to_asia=visit_to_asia,
                smoking=smoking,
                xray_result=xray_result,
                dyspnea=dyspnea,
            )

        # Debug prints (required by assignment): prompt + raw response.
        print(prompt)

        try:
            if self._local_tokenizer is None or self._local_model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                print(f"Loading local LLM tokenizer/model: {self._local_model_id}")
                self._local_tokenizer = AutoTokenizer.from_pretrained(
                    self._local_model_id, use_fast=True
                )
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    self._local_model_id,
                    low_cpu_mem_usage=True,
                )
                self._local_model.eval()
                print("Local LLM loaded successfully.")

            llm_input_text = prompt
            if hasattr(self._local_tokenizer, "apply_chat_template"):
                llm_input_text = self._local_tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Return ONLY valid JSON exactly in the form "
                                '{"disease": "TB|Cancer|Bronchitis", "probability": 0.0} '
                                "with no extra text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self._local_tokenizer(llm_input_text, return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]
            if self._local_tokenizer.pad_token_id is None:
                self._local_tokenizer.pad_token_id = self._local_tokenizer.eos_token_id

            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=self._local_max_new_tokens,
                do_sample=False,
                min_new_tokens=1,
                pad_token_id=self._local_tokenizer.pad_token_id,
                eos_token_id=self._local_tokenizer.eos_token_id,
            )
            gen_ids = output_ids[0][input_len:]
            raw_response = self._local_tokenizer.decode(
                gen_ids, skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"Error generating with local LLM: {e}")
            return ["TB", 0.0]

        print(raw_response)

        # Parse JSON safely using Python's json module.
        try:
            raw = raw_response.strip()
            parsed = json.loads(raw)
        except Exception:
            # If the model returns extra text, try to extract the first JSON object.
            match = re.search(r"\{[\s\S]*?\}", raw_response, flags=re.DOTALL)
            if not match:
                print("Error parsing JSON response: no JSON object found.")
                return ["TB", 0.0]
            parsed = json.loads(match.group(0))

        try:
            disease = parsed.get("disease")
            probability = float(parsed.get("probability"))
            if disease not in {"TB", "Cancer", "Bronchitis"}:
                raise ValueError("Invalid disease returned by LLM.")
            if not (0.0 <= probability <= 1.0):
                raise ValueError("Probability out of range.")

            result = [disease, round(probability, 3)]
            self._cache[cache_key] = tuple(result)
            return result
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return ["TB", 0.0]

