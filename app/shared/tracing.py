from uuid import uuid4


def new_response_id() -> str:
    return f"resp_{uuid4().hex}"


def new_trace_id() -> str:
    return f"trace_{uuid4().hex}"
