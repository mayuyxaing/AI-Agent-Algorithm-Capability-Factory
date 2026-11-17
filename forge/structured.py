"""Bounded correction of structured responses; invalid outputs remain in the audit."""


def structured_payload(response):
    """Unwrap exactly one sole answer object; never merge or invent fields."""
    if isinstance(response, dict) and set(response) == {"answer"} and isinstance(response["answer"], dict):
        return response["answer"]
    return response


def generation_payload(response):
    """Accept one unambiguous answer envelope without editing generated source."""
    value = response
    if isinstance(value, dict) and "answer" in value and set(value) <= {"answer", "explanation"} and isinstance(value["answer"], dict):
        outer = value
        value = outer["answer"]
        if "explanation" not in value and "explanation" in outer:
            value = dict(value, explanation=outer["explanation"])
    if not isinstance(value, dict) or not isinstance(value.get("code"), str) or not 1 <= len(value["code"]) <= 100_000:
        raise ValueError('Expected {"code": "complete Python source", "explanation": "..."}; code must be nonempty and under 100KB')
    if "explanation" in value and not isinstance(value["explanation"], str):
        raise ValueError("Generation explanation must be a string")
    return value


def checked_response(role, context, ask, validator, record, max_corrections=1):
    request = context
    for attempt in range(max_corrections + 1):
        result = ask(role, request)
        payload = structured_payload(result)
        if payload is not result:
            record(role + "_normalized", {"envelope": "answer", "depth": 1},
                   {"response": result, "normalized_response": payload, "fields_edited": False})
        try:
            validator(payload)
            return payload
        except ValueError as exc:
            record(role + "_rejected", {"attempt": attempt}, {"error": str(exc), "response": result})
            if attempt == max_corrections:
                raise ValueError(f"{role} contract rejected after {attempt + 1} responses: {exc}") from exc
            print(f"{role} 返回结构未通过校验，正根据错误反馈修正（{attempt + 1}/{max_corrections}）……", flush=True)
            request = dict(context, previous_response=result, validation_error=str(exc),
                           correction_instruction="Return a corrected JSON object that matches output_schema and all constraints. Do not relax constraints or invent citations.")
