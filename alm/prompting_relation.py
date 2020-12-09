""" Prompting function """

TEMPLATES = {
    'is-to-what': "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>",
    'is-to-as': "<subj-a> is to <obj-a> as <subj-b> is to <obj-b>",
    'rel-same': 'The relation between <subj-a> and <obj-a> is the same as the relation between <subj-b> and <obj-b>',
    'what-is-to': 'what <subj-a> is to <obj-a>, <subj-b> is to <obj-b>',
    'she-to-as': 'She explained to him that <subj-a> is to <obj-a> as <subj-b> is to <obj-b>.',
    'as-what-same': 'As I explained earlier, what <subj-a> is to <obj-a> is essentially the same as what <subj-b> is to <obj-b>.'
}

__all__ = ('prompting_relation', 'TEMPLATES')


def prompting_relation(subject_stem,
                       object_stem,
                       subject_analogy,
                       object_analogy,
                       template: str = None,
                       template_type: str = 'is-to-what'):
    """ to convert a SAT style analogy set into a natural sentence with a template """
    if template is None:
        template = TEMPLATES[template_type]
    assert "<subj-a>" in template and "<subj-b>" in template and "<obj-a>" in template and "<obj-b>" in template
    template = template.replace("<subj-a>", subject_stem).replace("<obj-a>", object_stem).\
        replace("<subj-b>", subject_analogy).replace("<obj-b>", object_analogy)
    return template
