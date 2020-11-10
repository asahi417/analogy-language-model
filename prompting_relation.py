""" Prompting function """

TEMPLATES = {
    'what-is': "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>",
    'as-is': "<subj-a> is to <obj-a> as <subj-b> is to <obj-b>"
}

__all__ = 'prompting_relation'


def prompting_relation(subject_stem,
                       object_stem,
                       subject_analogy,
                       object_analogy,
                       template_type: str = 'what-is'):
    template = TEMPLATES[template_type]
    template = template.replace("<subj-a>", subject_stem).replace("<obj-a>", object_stem).\
        replace("<subj-b>", subject_analogy).replace("<obj-b>", object_analogy)
    return template


if __name__ == '__main__':
    sample = {
        "stem": ["arid", "dry"],
        "answer": 0,
        "choice": [
            ["glacial", "cold"], ["coastal", "tidal"], ["damp", "muddy"], ["snowbound", "polar"], ["shallow", "deep"]],
        "prefix": "190 FROM REAL SATs"}
    prompt = prompting_relation(
        subject_stem=sample['stem'][0],
        object_stem=sample['stem'][1],
        subject_analogy=sample['choice'][0][1],
        object_analogy=sample['choice'][0][1])
    print(prompt)

