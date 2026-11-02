"""Rubric drafter: turns ONE section of a reference-solutions PDF into exam questions with
criteria (`huc rubric-draft`). The output is a draft for the teacher to review."""

from huc.domain.profile import CourseProfile
from huc.prompts.grading import bullets

RUBRIC_SYSTEM = """\
You turn a teacher's reference solutions into an exam definition with a grading rubric.
You receive the rendered pages of a solutions document and its extracted text. Use ONLY \
the section titled `{section}`: the first and last pages may also show the end of the \
previous section or the start of the next one.

Return one question per problem; a problem with sub-parts a), b), c) becomes one \
question per sub-part (ids `p1a`, `p1b`, numbers "1a", "1b"); otherwise ids `p1`, `p2` \
and numbers "1", "2".

For each question:
- `prompt`: the problem statement, verbatim, in its original language. For a sub-part, \
include the shared statement it needs (definitions, data) so it reads on its own.
- `reference_answer`: the reference solution condensed to its key steps and result.
- `subject`: {subjects}
- `skill`: what the problem asks to do, as a short verb phrase (e.g. "Demostrar por \
contradicción", "Determinar y justificar").
- `constraints`: restrictions the statement explicitly imposes on the answer (e.g. "A \
partir de los axiomas de cuerpo"); empty if there are none. Never invent them.
- `criteria`: 1 to {max_criteria} criteria{points_rule}. Each criterion is one essential \
thing the answer must establish, stated as WHAT must be shown, not how, so that any \
correct method can earn it. Give integer `max_points` and one level per score from \
`max_points` down to 0, each with a short descriptor; intermediate levels reward a \
correct but incomplete idea. Criteria must not overlap (no double penalization).
- Criterion ids: short snake_case, unique within the question.

Write math on a single line with Unicode symbols: fractions as (x+y)/2, powers as x^2, \
subscripts as x_n, roots as √(x+1), sums as Σ_(j=1)^n a_j, and symbols such as ∈ ⊆ ∀ ∃ \
≤ ≥ ≠ ⇒ ⇔ △ ∠. Write criteria, levels and skills in the language of the solutions.
{course}{guidelines}"""


def rubric_system(section: str, profile: CourseProfile | None = None) -> str:
    profile = profile or CourseProfile()
    policy = profile.rubric
    if profile.subjects:
        subjects = "one of " + ", ".join(f"`{s.id}` ({s.name})" for s in profile.subjects)
        subjects += ", or null if none fits."
    else:
        subjects = "null."
    points_rule = (
        f" whose `max_points` add up to {policy.points_per_question:g}"
        if policy.points_per_question
        else ""
    )
    course = ""
    if profile.name or profile.description:
        course = f"\n## Course: {profile.name}\n{profile.description.strip()}\n"
    guidelines = f"\n## Rubric guidelines\n{bullets(policy.guidelines)}\n" if policy.guidelines else ""
    return RUBRIC_SYSTEM.format(
        section=section,
        subjects=subjects,
        max_criteria=policy.max_criteria,
        points_rule=points_rule,
        course=course,
        guidelines=guidelines,
    )


def section_text_block(text: str) -> dict:
    return {"type": "text", "text": f"<extracted_text>\n{text}\n</extracted_text>"}
