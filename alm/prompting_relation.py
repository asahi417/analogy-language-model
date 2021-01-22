""" Prompting function """

TEMPLATES = {
    'is-to-what': "<subj-a> is to <obj-a> what <subj-b> is to <obj-b>",
    'is-to-as': "<subj-a> is to <obj-a> as <subj-b> is to <obj-b>",
    'rel-same': 'The relation between <subj-a> and <obj-a> is the same as the relation between <subj-b> and <obj-b>',
    'what-is-to': 'what <subj-a> is to <obj-a>, <subj-b> is to <obj-b>',
    'she-to-as': 'She explained to him that <subj-a> is to <obj-a> as <subj-b> is to <obj-b>.',
    'as-what-same': 'As I explained earlier, what <subj-a> is to <obj-a> is essentially the same as what <subj-b> is'
                    'to <obj-b>.'
}
__all__ = ('prompting_relation', 'TEMPLATES')


def check_position(text, positions, tokens):
    for p, t in zip(positions, tokens):
        assert text[p[0]: p[1]] == t, '{} != {}'.format(text[p[0]: p[1]], t)


def prompting_relation(relation_words, template_type: str = 'is-to-what'):
    """ to convert a SAT style analogy set into a natural sentence with a template """
    template = TEMPLATES[template_type]
    subject_a, object_a, subject_b, object_b = relation_words
    position = []
    for i, m in zip(['<subj-a>', '<obj-a>', '<subj-b>', '<obj-b>'], [subject_a, object_a, subject_b, object_b]):
        position += [[len(template.split(i)[0]), len(template.split(i)[0]) + len(m)]]
        template = template.replace(i, m)
    check_position(template, position, [subject_a, object_a, subject_b, object_b])
    return template, position
