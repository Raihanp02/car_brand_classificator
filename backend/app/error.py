# app/errors.py
class AppError(Exception):
    def __init__(self, code: str, message: str = "", **details):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.details = details

class InferenceError(AppError):
    """Inference failed because of our side (broken index, bad model output).

    Never caused by client input.
    """

    code = "inference_failed"
    