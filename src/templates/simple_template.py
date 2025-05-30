from src.templates.utils import append_proper_article

simple_template = [
    lambda c: f"a photo of a {c}."
]

fashion_template = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"this is a {c}.",
    lambda c: f"a beautiful {c}.",
    lambda c: f"an elegant {c}.",
    lambda c: f"a stylish {c}.",
    lambda c: f"a fashionable {c}.",
    lambda c: f"a {c} garment.",
    lambda c: f"traditional {c}.",
    lambda c: f"handwoven {c}.",
    lambda c: f"ethnic {c}.",
]