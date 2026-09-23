from __future__ import annotations

import re


_ENGLISH_RE = re.compile(r"[A-Za-z]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_PROMPT_LEAK_RE = re.compile(r"<\/?think>|system:|assistant:|user:", re.IGNORECASE)
_SECURITY_CLAIM_RE = re.compile(
    r"security|secure|safety|privacy|breach|hack|encryption|fraud|scam|theft|stolen|identity",
    re.IGNORECASE,
)
_NON_SECURITY_DENIAL_RE = re.compile(
    r"non-security|not (?:a )?security|unrelated to security", re.IGNORECASE
)


def build_explanation_messages(review: str, predicted_label: str) -> list[dict[str, str]]:
    if predicted_label not in {"security", "non-security"}:
        raise ValueError(f"Unsupported label: {predicted_label}")
    if predicted_label == "security":
        constraint = (
            "Explain the specific security concern present in the review, such as privacy, "
            "authentication, fraud, or an unauthorized transaction. Do not add a risk that the review does not state."
        )
    else:
        constraint = (
            "Explain only the operational, functional, performance, usability, language, advertising, "
            "update, or customer-support point stated in the review. Do not claim or imply any security, safety, or privacy impact."
        )
    return [
        {
            "role": "system",
            "content": (
                "You classify mobile-app user reviews. The supplied classification is final, and the explanation "
                "must agree with it without debating or changing it. Respond in English only with one concise sentence, "
                "without reasoning steps, headings, or extra notes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Review: {review}\nFinal classification: {predicted_label}\n"
                f"{constraint} Use only information supported by the review."
            ),
        },
    ]


def fallback_explanation(review: str, predicted_label: str) -> str:
    text = " ".join(str(review).split())
    if predicted_label == "security":
        if re.search(r"رمز التحقق|تسجيل الدخول|كلمة المرور|بصم|مصادق", text):
            return "The review reports a verification or login problem involving access to the account."
        if re.search(r"خصوص|تجسس|تشفير|بيانات|رسائل", text):
            return "The review describes spying, exposed data, or missing encryption, creating a clear privacy concern."
        if re.search(r"احتيال|نصب", text):
            return "The review explicitly alleges fraud or a scam affecting the user."
        if re.search(r"سحب|خصم|مبلغ|بطاق", text):
            return "The review describes a charge, withdrawal, or card action without the user's clear authorization."
        return "The review describes a direct concern involving the user's account, data, or protection."

    if re.search(r"تحديث|اصدار|إصدار", text):
        return "The review says the app stopped opening after the update, indicating an update-related functional failure."
    if re.search(r"الدعم|خدمة العملاء", text):
        return "The review combines an app malfunction with poor customer support."
    if re.search(r"اعلان|إعلان|دعاي", text):
        return "The review complains that intrusive advertising harms the user experience."
    if re.search(r"اللغة العربية|لغة عربية|العربية", text):
        return "The review complains that the app does not support Arabic."
    if re.search(r"بطي|سريع|تعليق|يعلق", text):
        return "The review describes slowness or freezing, indicating a performance problem."
    if re.search(r"سهل|سهولة", text):
        return "The review praises the app for being easy to use."
    if re.search(r"لا\s*يعمل|لا\s*يفتح|تعطل|عطل", text):
        return "The review says the app does not work or open, indicating a functional problem."
    return "The review concerns the app's functionality or user experience."


def finalize_explanation(review: str, predicted_label: str, candidate: str) -> str:
    explanation = " ".join(str(candidate).strip().split())
    review_text = " ".join(str(review).split())
    invalid = (
        not explanation
        or not _ENGLISH_RE.search(explanation)
        or bool(_ARABIC_RE.search(explanation))
        or bool(_PROMPT_LEAK_RE.search(explanation))
    )
    if predicted_label == "non-security" and _SECURITY_CLAIM_RE.search(explanation):
        invalid = True
    if predicted_label == "non-security" and re.search(r"تحديث|اصدار|إصدار", review_text):
        if not re.search(r"functional|update|operation|performance|problem|failure", explanation, re.IGNORECASE):
            invalid = True
    if predicted_label == "non-security" and re.search(r"الدعم|خدمة العملاء", review_text):
        has_support = bool(re.search(r"customer support|support service", explanation, re.IGNORECASE))
        review_has_malfunction = bool(re.search(r"لا\s*يعمل|لا\s*يفتح|تعطل|عطل", review_text))
        has_malfunction = bool(re.search(r"app (?:does not|doesn't) work|functional|malfunction|fails? to (?:work|open)", explanation, re.IGNORECASE))
        if not has_support or (review_has_malfunction and not has_malfunction):
            invalid = True
    if predicted_label == "security" and _NON_SECURITY_DENIAL_RE.search(explanation):
        invalid = True
    return fallback_explanation(review, predicted_label) if invalid else explanation
