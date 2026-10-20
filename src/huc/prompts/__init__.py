"""Agent prompts: one module per agent, containing only prompt text and render helpers.

Caching rule: everything that is identical for all students of an exam goes in the
system prompt (cacheable prefix). The only variable part, the student's answer,
goes in the user message. Never put dates, student IDs or other per-run data in a
system prompt.
"""
